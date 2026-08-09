//! Gemma 4's own parameter blocks, replicated exactly from the kernel
//! headers. Each is the ABI of one small kernel; the family's larger
//! blocks (RMS, router, mixture movers) are the shared ones.

/// `vnorm_single_row`'s params: RMS with NO learnable weight, applied
/// to V before the KV write. Distinct from `RmsParams`, which always
/// has one.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VNormParams {
    /// The norm epsilon.
    pub eps: f32,
    /// The per-head axis.
    pub axis_size: u32,
}
const _: () = assert!(size_of::<VNormParams>() == 8);

/// `geglu_tanh`'s params: the flat element count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GegluParams {
    /// Elements the dispatch covers.
    pub n: u32,
}
const _: () = assert!(size_of::<GegluParams>() == 4);

/// `layer_scalar_mul`'s params.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerScalarParams {
    /// Elements the dispatch covers.
    pub n: u32,
}
const _: () = assert!(size_of::<LayerScalarParams>() == 4);

/// `ple_combine`'s params: `out = (proj + token) · inv_sqrt2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PleCombineParams {
    /// 1/√2, stated rather than computed on device.
    pub inv_sqrt2: f32,
    /// Elements the dispatch covers (`n_layers × ple_dim`).
    pub n: u32,
}
const _: () = assert!(size_of::<PleCombineParams>() == 8);

/// `logit_softcap`'s params: `out = cap · tanh(logits / cap)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftcapParams {
    /// The cap.
    pub cap: f32,
    /// Elements the dispatch covers.
    pub n: u32,
}
const _: () = assert!(size_of::<SoftcapParams>() == 8);
