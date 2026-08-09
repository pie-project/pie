//! `WeightView` — how every kernel is handed a weight.
//!
//! Port of `kernels-cuda/csrc/src/weight_view.hpp` and of `make_weight_view`
//! from `driver-cuda/csrc/src/model/llama_like/qwen3.hpp`.
//!
//! # This one crosses the ABI, so the layout is the contract
//!
//! `WeightView` is passed to kernel launchers **by value**. Everything else
//! ported so far lives entirely on the Rust side of the boundary and only has
//! to *behave* like its C++ original; this one has to *be laid out* like it,
//! because the bytes are handed to a C++ function that reads them back by
//! offset.
//!
//! Nothing about a mirror that is wrong by four bytes looks wrong. The GEMM
//! would read `scale_data` out of the padding after `dtype` and dereference
//! it. So the layout is not asserted by reading the header and agreeing with
//! it — `tests/weight_view_parity.rs` pins `size_of`, `align_of` and all ten
//! field offsets against numbers the C++ printed with `offsetof`.
//!
//! Measured, from that oracle: 72 bytes, 8-aligned, with 7 bytes of padding
//! after `dtype`, 7 after `scale_dtype`, and 4 after `quant_kind`. Rust's
//! `repr(C)` uses the same rules, so this falls out of declaring the fields in
//! the same order with the same types — but "falls out of" is a claim, and the
//! test is what makes it one that has been checked.
//!
//! # Two null-looking things that are not bugs
//!
//! Both have the shape of the `workspace_bytes` bug (`.wiki/kernel-refactor`
//! §8) — a field one path fills and another leaves at zero — and both were
//! run down to the mechanism that makes them safe rather than reported.
//!
//! 1. **[`WeightView::raw`] leaves `nbytes` at 0.** The only reader of
//!    `nbytes` is `validate_quant_weight_view` (`gemm/gemm.cpp:1318`), which
//!    is gated behind `scale_data != nullptr` — and `raw` leaves that null
//!    too, so a raw view throws "quant scale data is null" before the size
//!    check can compare against a bogus zero. That is a load-bearing agreement
//!    between two functions in different files, which is why the parity
//!    transcript records every field of every factory rather than the ones
//!    that looked interesting.
//!
//! 2. **A [`QuantMeta`] with no scale produces a view byte-identical to the
//!    unquantized one**, because `scale_data == nullptr` *is* the dispatcher's
//!    "bf16 path" signal. Reading quantized bytes as bf16 would be silent
//!    garbage. It cannot happen: `WeightStore::validate_quant_metadata`
//!    (`model/weight_store.cpp:230`) runs at load time and rejects any
//!    metadata whose scale is not a registered tensor, and
//!    `owns_tensor_handle(nullptr)` is `false`. No `QuantMeta` reachable from
//!    a loaded model has a null scale.

use core::ffi::c_void;

use crate::dtype::DType;

/// How a quantized weight's scales are laid out.
///
/// `#[repr(i32)]` because the C++ `enum class Kind` has no fixed underlying
/// type and therefore gets `int` — 4 bytes, signed. The oracle measures this
/// rather than assuming it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum QuantKind {
    /// One scale for the whole tensor.
    #[default]
    PerTensor = 0,
    /// One scale per output channel.
    PerChannel = 1,
    /// One scale per group of `group_size` elements along the reduction axis.
    PerGroup = 2,
}

/// Per-weight quantization metadata, as the driver holds it.
///
/// Port of `kernels-cuda/csrc/src/quant_meta.hpp`. This one does **not** cross
/// the ABI — it owns `std::string` names in C++ — so it is a plain Rust
/// struct. Only its *contents* travel, flattened into a [`WeightView`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantMeta {
    /// Scale layout.
    pub kind: QuantKind,
    /// The scale tensor's device pointer, dtype and element count.
    ///
    /// `None` is representable here because the C++ field is a raw pointer
    /// that starts null, but see the module docs: no metadata reachable from a
    /// loaded model has one.
    pub scale: Option<TensorRef>,
    /// The zero-point tensor, for asymmetric quantization.
    pub zero_point: Option<TensorRef>,
    /// Elements per scale group, when `kind` is [`QuantKind::PerGroup`].
    pub group_size: i32,
    /// Which axis the per-channel scales index.
    pub channel_axis: i32,
}

/// The three things a [`WeightView`] needs from a tensor.
///
/// Stands in for `const DeviceTensor&` at the factories. The C++ deliberately
/// does not store the tensor pointer — bind functions hand it raw pointers
/// into fused expert tables that have no `DeviceTensor` wrapper — so carrying
/// `(data, dtype, nbytes)` is what covers both cases, and this type says that
/// out loud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRef {
    /// Device pointer to the first element.
    pub data: *const c_void,
    /// Element type.
    pub dtype: DType,
    /// Size of the backing allocation.
    pub nbytes: usize,
    /// Element count, used only for a scale tensor's `scale_numel`.
    pub numel: usize,
}

impl TensorRef {
    /// A tensor reference from its three device facts.
    #[must_use]
    pub const fn new(data: *const c_void, dtype: DType, nbytes: usize, numel: usize) -> Self {
        Self {
            data,
            dtype,
            nbytes,
            numel,
        }
    }
}

/// Lightweight reference to a weight tensor plus optional quantization
/// metadata, threaded through the GEMM dispatcher.
///
/// **`#[repr(C)]` is load-bearing**: this is handed to C++ launchers by value.
/// The field order below is the C++ declaration order and must not be
/// rearranged, however much nicer a packed order would be — `tests/
/// weight_view_parity.rs` pins every offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WeightView {
    /// Device pointer to the weight bytes.
    pub data: *const c_void,
    /// Element type of `data`.
    pub dtype: DType,
    /// Size of the weight allocation.
    ///
    /// Read only by the quantized path's size guard; [`Self::raw`] leaves it
    /// zero. See the module docs for why that is safe.
    pub nbytes: usize,
    /// Device pointer to the scales.
    ///
    /// **Null means "no quant — bf16 path".** This is the dispatcher's
    /// discriminator, not `dtype` and not `quant_kind`.
    pub scale_data: *const c_void,
    /// Element type of the scales.
    pub scale_dtype: DType,
    /// Number of scale elements.
    pub scale_numel: usize,
    /// Scale layout, for the per-channel and per-group paths.
    pub quant_kind: QuantKind,
    /// Device pointer to the zero points, for asymmetric quantization.
    pub zero_point_data: *const c_void,
    /// Elements per scale group.
    pub group_size: i32,
    /// Which axis the per-channel scales index.
    pub channel_axis: i32,
}

impl Default for WeightView {
    /// Matches the C++ `WeightView() = default` with its member initialisers:
    /// null pointers, `BF16` for `dtype`, **`FP32` for `scale_dtype`**, and
    /// zero for everything else. The asymmetry between the two dtype defaults
    /// is in the original and is preserved deliberately.
    fn default() -> Self {
        Self {
            data: core::ptr::null(),
            dtype: DType::Bf16,
            nbytes: 0,
            scale_data: core::ptr::null(),
            scale_dtype: DType::Fp32,
            scale_numel: 0,
            quant_kind: QuantKind::PerTensor,
            zero_point_data: core::ptr::null(),
            group_size: 0,
            channel_axis: 0,
        }
    }
}

impl WeightView {
    /// The unquantized view: what the C++ implicit
    /// `WeightView(const DeviceTensor&)` produces.
    ///
    /// Not a `From` impl. The C++ conversion is implicit and that is exactly
    /// what makes `lm_head_argmax_supported` subtle — its callers all reach it
    /// through this constructor, which hard-codes `scale_data = nullptr`, so
    /// in practice only the dtype discriminates. A named constructor makes the
    /// same call sites say which view they meant.
    #[must_use]
    pub const fn plain(weight: TensorRef) -> Self {
        Self {
            data: weight.data,
            dtype: weight.dtype,
            nbytes: weight.nbytes,
            scale_data: core::ptr::null(),
            scale_dtype: DType::Fp32,
            scale_numel: 0,
            quant_kind: QuantKind::PerTensor,
            zero_point_data: core::ptr::null(),
            group_size: 0,
            channel_axis: 0,
        }
    }

    /// Raw pointer plus dtype, for buffers with no tensor handle
    /// (deinterleaved MoE scratch, expert pointer arrays).
    ///
    /// Leaves `nbytes` at 0, as the C++ does. Safe only because the sole
    /// reader of `nbytes` is behind a `scale_data != nullptr` gate this cannot
    /// pass; see the module docs.
    #[must_use]
    pub const fn raw(data: *const c_void, dtype: DType) -> Self {
        Self {
            data,
            dtype,
            nbytes: 0,
            scale_data: core::ptr::null(),
            scale_dtype: DType::Fp32,
            scale_numel: 0,
            quant_kind: QuantKind::PerTensor,
            zero_point_data: core::ptr::null(),
            group_size: 0,
            channel_axis: 0,
        }
    }

    /// A quantized weight: a weight tensor tied to a [`QuantMeta`] snapshot.
    ///
    /// When `meta.scale` is `None` this produces a view byte-identical to
    /// [`Self::plain`], which is the C++ behaviour. That is unreachable from a
    /// loaded model — see the module docs — and [`Self::would_silently_degrade`]
    /// is how a caller can check rather than assume.
    #[must_use]
    pub fn quantized(weight: TensorRef, meta: &QuantMeta) -> Self {
        Self {
            data: weight.data,
            dtype: weight.dtype,
            nbytes: weight.nbytes,
            scale_data: meta.scale.map_or(core::ptr::null(), |s| s.data),
            scale_dtype: meta.scale.map_or(DType::Fp32, |s| s.dtype),
            scale_numel: meta.scale.map_or(0, |s| s.numel),
            quant_kind: meta.kind,
            zero_point_data: meta.zero_point.map_or(core::ptr::null(), |z| z.data),
            group_size: meta.group_size,
            channel_axis: meta.channel_axis,
        }
    }

    /// Marlin-packed MXFP4: E2M1 values with E8M0 block scales.
    ///
    /// Overrides the weight's own dtype with [`DType::Mxfp4Packed`] and
    /// hard-codes the group size to 32 and `channel_axis` to 0, because that
    /// is what the packing means. A caller that has a real `channel_axis`
    /// (glm5's expert path) sets it afterwards.
    #[must_use]
    pub const fn mxfp4_marlin(weight: TensorRef, scale: TensorRef) -> Self {
        Self {
            data: weight.data,
            dtype: DType::Mxfp4Packed,
            nbytes: weight.nbytes,
            scale_data: scale.data,
            scale_dtype: DType::Uint8,
            scale_numel: scale.numel,
            quant_kind: QuantKind::PerGroup,
            zero_point_data: core::ptr::null(),
            group_size: 32,
            channel_axis: 0,
        }
    }

    /// Whether this view claims a quantized layout but carries no scales, and
    /// would therefore be dispatched down the bf16 path.
    ///
    /// The condition the C++ has no way to express and no place to ask. It is
    /// unreachable through the loader — `validate_quant_metadata` rejects it —
    /// but "unreachable because something two crates away checks" is worth
    /// being able to assert at the point of use rather than only in a comment.
    #[must_use]
    pub fn would_silently_degrade(&self) -> bool {
        self.scale_data.is_null() && self.quant_kind != QuantKind::PerTensor
    }

    /// Whether the dispatcher will treat this as an unquantized bf16 weight.
    #[must_use]
    pub fn is_bf16_path(&self) -> bool {
        self.scale_data.is_null()
    }
}

/// Pick the right view for a `(weight, optional quant metadata)` pair.
///
/// Port of `make_weight_view` in `model/llama_like/qwen3.hpp`, which the
/// generated forward bodies call 1,068 times. The discriminator is whether the
/// optional is engaged, **not** whether it carries a scale — an engaged but
/// empty one takes the quantized branch and comes out looking unquantized.
#[must_use]
pub fn make_weight_view(weight: TensorRef, meta: Option<&QuantMeta>) -> WeightView {
    match meta {
        Some(m) => WeightView::quantized(weight, m),
        None => WeightView::plain(weight),
    }
}

/// A weight the trace names but the bind step never resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingWeight<'a>(pub &'a str);

impl MissingWeight<'_> {
    /// The C++ `what()` string, so a parity transcript can compare messages.
    #[must_use]
    pub fn cpp_message(&self) -> String {
        format!(
            "declared forward: weight '{}' is named by the trace but not bound",
            self.0
        )
    }
}

/// The generated bodies' `require(ptr, "name")`: 5,703 calls.
///
/// The C++ throws; here it is a `Result`, so a caller that forgets to handle
/// the missing weight does not compile rather than unwinding through a graph
/// capture.
pub fn require<T>(tensor: Option<T>, name: &str) -> Result<T, MissingWeight<'_>> {
    tensor.ok_or(MissingWeight(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(n: usize) -> *const c_void {
        n as *const c_void
    }

    fn weight() -> TensorRef {
        TensorRef::new(ptr(0x1000), DType::Fp8E4M3, 8192, 8192)
    }

    fn scale() -> TensorRef {
        TensorRef::new(ptr(0x2000), DType::Fp32, 512, 128)
    }

    #[test]
    fn the_layout_is_the_one_the_cpp_launchers_read() {
        // Pinned against `offsetof` output from the real header; see
        // tests/weight_view_parity.rs for the full transcript.
        assert_eq!(size_of::<WeightView>(), 72);
        assert_eq!(align_of::<WeightView>(), 8);
        assert_eq!(size_of::<QuantKind>(), 4);
        assert_eq!(size_of::<DType>(), 1);
    }

    #[test]
    fn raw_leaves_nbytes_zero_but_cannot_reach_the_reader_of_nbytes() {
        let v = WeightView::raw(ptr(0x3000), DType::Bf16);
        assert_eq!(v.nbytes, 0, "matches the C++, which also leaves it unset");
        assert!(
            v.is_bf16_path(),
            "which is what keeps the zero out of the size guard"
        );
        assert!(!v.would_silently_degrade());
    }

    #[test]
    fn a_quant_meta_with_no_scale_is_indistinguishable_from_an_unquantized_view() {
        // The behaviour the loader's validate_quant_metadata exists to make
        // unreachable. Pinned so that a future change to `quantized` which
        // "fixed" it would be a deliberate divergence from the C++, not a
        // silent one.
        let empty = QuantMeta::default();
        assert_eq!(
            WeightView::quantized(weight(), &empty),
            WeightView::plain(weight())
        );
        assert_eq!(
            make_weight_view(weight(), Some(&empty)),
            make_weight_view(weight(), None)
        );
    }

    #[test]
    fn a_quantized_view_that_lost_its_scales_can_be_asked_about() {
        let mut meta = QuantMeta {
            kind: QuantKind::PerChannel,
            scale: Some(scale()),
            ..QuantMeta::default()
        };
        assert!(!WeightView::quantized(weight(), &meta).would_silently_degrade());
        meta.scale = None;
        assert!(
            WeightView::quantized(weight(), &meta).would_silently_degrade(),
            "a per-channel view with no scales is the condition worth naming"
        );
        // PerTensor with no scale is just the bf16 path and is not a defect.
        meta.kind = QuantKind::PerTensor;
        assert!(!WeightView::quantized(weight(), &meta).would_silently_degrade());
    }

    #[test]
    fn the_marlin_factory_overrides_the_weights_own_dtype() {
        let v = WeightView::mxfp4_marlin(weight(), scale());
        assert_eq!(
            v.dtype,
            DType::Mxfp4Packed,
            "the packing, not the tensor, decides"
        );
        assert_eq!(v.scale_dtype, DType::Uint8, "E8M0 scales are uint8");
        assert_eq!(v.group_size, 32);
        assert_eq!(v.channel_axis, 0);
        assert_eq!(v.nbytes, weight().nbytes, "unlike raw, this one sets it");
    }

    #[test]
    fn the_default_scale_dtype_is_fp32_not_bf16() {
        // The asymmetry is in the C++ member initialisers and is easy to
        // "tidy" away.
        let v = WeightView::default();
        assert_eq!(v.dtype, DType::Bf16);
        assert_eq!(v.scale_dtype, DType::Fp32);
    }

    #[test]
    fn require_names_the_weight_the_trace_asked_for() {
        assert_eq!(require(Some(7u32), "layer.0.q_proj"), Ok(7));
        let err = require::<u32>(None, "layer.0.q_proj").unwrap_err();
        assert_eq!(
            err.cpp_message(),
            "declared forward: weight 'layer.0.q_proj' is named by the trace but not bound"
        );
    }
}
