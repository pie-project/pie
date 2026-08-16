//! `WeightView` — how every kernel would be handed a weight. No production
//! caller, deliberately: `tests/weight_view_parity.rs` exercises it against a golden hash for ABI/behavioural parity with the C++ type.
//! Two safe nulls: `raw` leaves `nbytes` 0 without setting `scale_data`; a scaleless `QuantMeta` equals the plain bf16 view.

use core::ffi::c_void;

use crate::dtype::DType;

/// How a quantized weight's scales are laid out; `#[repr(i32)]` because the C++ `enum class Kind` is a plain `int`.
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

/// Per-weight quantization metadata, as the driver holds it; unlike [`WeightView`] it does not cross the ABI.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantMeta {
    /// Scale layout.
    pub kind: QuantKind,
    /// The scale tensor's device pointer, dtype and element count.
    pub scale: Option<TensorRef>,
    /// The zero-point tensor, for asymmetric quantization.
    pub zero_point: Option<TensorRef>,
    /// Elements per scale group, when `kind` is [`QuantKind::PerGroup`].
    pub group_size: i32,
    /// Which axis the per-channel scales index.
    pub channel_axis: i32,
}

/// The three things a [`WeightView`] needs from a tensor: stands in for `const DeviceTensor&`, but also covers wrapper-less raw pointers.
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

/// Lightweight reference to a weight tensor plus optional quantization metadata, threaded through the GEMM dispatcher; `#[repr(C)]`, do not rearrange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WeightView {
    /// Device pointer to the weight bytes.
    pub data: *const c_void,
    /// Element type of `data`.
    pub dtype: DType,
    /// Size of the weight allocation; [`Self::raw`] leaves it zero (module docs).
    pub nbytes: usize,
    /// Device pointer to the scales; null (not `dtype`/`quant_kind`) is the dispatcher's bf16-path discriminator.
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
    /// Matches the C++ `WeightView() = default`: null pointers, `BF16` `dtype` but `FP32` `scale_dtype` (asymmetry kept deliberately), zero else.
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
    /// The unquantized view: the C++ implicit `WeightView(const DeviceTensor&)`; a named ctor (not `From`) so call sites say which view they meant.
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

    /// Raw pointer plus dtype, for buffers with no tensor handle (MoE scratch, expert pointer arrays); leaves `nbytes` 0 — safe per the module docs.
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

    /// A quantized weight tied to a [`QuantMeta`] snapshot; a `None` scale yields a view identical to [`Self::plain`] (module docs).
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

    /// Marlin-packed MXFP4 (E2M1 + E8M0 scales): forces group size 32 and `channel_axis` 0; glm5's experts override the axis after.
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

    /// Whether this view claims quantized but carries no scales, so would silently dispatch bf16; unreachable through the loader.
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

/// Picks the right view for a `(weight, optional quant metadata)` pair: discriminated by whether the optional is engaged, not whether it has a scale.
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

/// The generated bodies' `require(ptr, "name")`: a `Result`, not the C++ throw, so a missing weight fails to compile instead of unwinding a capture.
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
        // Pinned against `offsetof`; see tests/weight_view_parity.rs.
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
        // Unreachable via validate_quant_metadata; pinned against a C++ divergence.
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
        // The asymmetry is in the C++ member initialisers and easy to "tidy".
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
