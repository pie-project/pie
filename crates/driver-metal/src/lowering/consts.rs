//! The geometry-derived constants every dispatch binds, and the kernel
//! param structs they travel in.
//!
//! The ported kernels take geometry constants — norm epsilons, projection
//! K/N, rope base, SDPA strides — that no other lane binds: not the weight
//! registry, not the scratch schedule, not the KV/state/IO binds. The
//! argument tables are address-only (no `setBytes`), so an unbound
//! `constant&` argument is a GPU fault, not a compile error; this module is
//! the one place those values are derived.
//!
//! Every struct layout here is replicated EXACTLY from the kernel headers
//! (`kernels-metal/kernels/{norm/rms_params.h, moe/params.h,
//! ssm/gdn_params.h}`), with size asserts standing in for the C++
//! `static_assert`s.

use crate::lowering::abi::Kernel;
use crate::batch::geometry::DecodeGeometry;

/// A projection's in/out vector lengths, from geometry — matching the
/// staged weight shapes. The split-K rule needs K, not just the output
/// width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KN {
    /// The input width.
    pub k: u32,
    /// The output width; zero means "not a matvec kind".
    pub n: u32,
}

/// `KN` per matvec kind.
///
/// Asked of THIS model's geometry, never a default one: the C++ once asked
/// a default-constructed geometry, which worked only because every dense
/// width is nonzero whatever the numbers are. A routed projection's is not
/// — `moe_intermediate` and `n_experts` are zero in a default geometry, so
/// the mixture's matvecs answered "not a matvec", skipped the K/N binding
/// entirely, and ran against whatever the pool held at those ordinals.
///
/// `GdnInA`/`GdnInB` are quantized like every other projection in the
/// layer. They were once bound as dense bf16 — what a Qwen3-Next preview
/// repack shipped and what no released checkpoint ships — so reading
/// `in_proj_a.weight` as bf16 read packed nibbles as floats: NaN in the
/// first four output channels, small plausible wrong numbers in the other
/// twelve, and the model produced token 0 forever.
#[must_use]
pub fn qmv_kn(kind: Kernel, g: &DecodeGeometry) -> KN {
    let h = g.hidden;
    let q_wide = 2 * g.n_q_heads * g.head_dim;
    let kv_dim = g.n_kv_heads * g.head_dim;
    let q_dim = g.n_q_heads * g.head_dim;
    let kn = |k, n| KN { k, n };
    match kind {
        Kernel::QmvIn => kn(h, g.gdn_conv_dim),
        Kernel::QmvInZ => kn(h, g.gdn_v_total),
        Kernel::GdnInA | Kernel::GdnInB => kn(h, g.gdn_v_heads),
        Kernel::QmvOut => kn(g.gdn_v_total, h),
        Kernel::QmvQ => kn(h, q_wide),
        Kernel::QmvK | Kernel::QmvV => kn(h, kv_dim),
        Kernel::QmvO => kn(q_dim, h),
        Kernel::QmvGate | Kernel::QmvUp => kn(h, g.intermediate),
        Kernel::QmvDown => kn(g.intermediate, h),
        Kernel::QmvLmHead | Kernel::LmHeadUntied => kn(h, g.vocab),
        // The mixture: the router is an ordinary matvec into one logit per
        // expert; the expert projections have a K and an N like any other —
        // what makes them routed is which weight slice a row reads, not
        // their shape.
        Kernel::Router => kn(h, g.n_experts),
        Kernel::ExpertGate | Kernel::ExpertUp => kn(h, g.moe_intermediate),
        Kernel::ExpertDown => kn(g.moe_intermediate, h),
        // The shared expert: ordinary dense shapes; the gate is
        // `hidden -> 1`, a matvec only in the sense that everything with a
        // K and an N is.
        Kernel::SharedGate | Kernel::SharedUp => kn(h, g.shared_intermediate),
        Kernel::SharedDown => kn(g.shared_intermediate, h),
        Kernel::SharedGateProj => kn(h, 1),
        _ => kn(0, 0),
    }
}

/// Whether a kind is a matvec, asked of this model's geometry.
#[must_use]
pub fn is_qmv(kind: Kernel, g: &DecodeGeometry) -> bool {
    qmv_kn(kind, g).n != 0
}

/// The three projections that read one expert's slice per row.
#[must_use]
pub fn is_routed(kind: Kernel) -> bool {
    matches!(
        kind,
        Kernel::ExpertGate | Kernel::ExpertUp | Kernel::ExpertDown
    )
}

/// `norm/rms_params.h::RmsParams`. The gain is the RAW weight: qwen3.5 is
/// not Gemma — its norm weights are absolute (input_layernorm averages
/// 1.24, `model.norm` 4.31 on the 0.8B checkpoint), not the `w - 1`
/// offsets a `1 + w` gain expects. A `1 + w` gain here is finite and
/// quiet, surviving as a ~80% per-norm error the residual stream
/// compounds.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsParams {
    /// The epsilon.
    pub eps: f32,
    /// The row width.
    pub axis_size: u32,
    /// Elements between adjacent weight entries.
    pub w_stride: u32,
    /// Whether the kernel adds one to the weight (Gemma); qwen3.5 sets 1
    /// for `plus_one`… the C++ binds `{eps, hidden, 1, 0, 1.0}`: w_stride
    /// 1, plus_one 0, gain 1.
    pub plus_one: u32,
    /// A constant multiplier on the gain (gemma4's router uses
    /// `hidden^-0.5`).
    pub gain: f32,
}
const _: () = assert!(size_of::<RmsParams>() == 20);

/// `norm/rms_params.h::GatedRmsParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedRmsParams {
    /// The epsilon.
    pub eps: f32,
    /// The value width one head normalizes over.
    pub vd: u32,
}
const _: () = assert!(size_of::<GatedRmsParams>() == 8);

/// `ssm/gdn_params.h::GdnCoreParams` — shared by every GDN core variant.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)] // field names are the kernel header's, one-to-one
pub struct GdnCoreParams {
    pub dk: i32,
    pub dv: i32,
    pub hk: i32,
    pub hv: i32,
    pub conv_dim: i32,
    pub kc: i32,
    pub q_off: i32,
    pub k_off: i32,
    pub v_off: i32,
    pub eps: f32,
    pub inv_sqrt_dk: f32,
}
const _: () = assert!(size_of::<GdnCoreParams>() == 44);

/// The GDN params for this geometry.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn gdn_core_params(g: &DecodeGeometry) -> GdnCoreParams {
    GdnCoreParams {
        dk: g.gdn_k_dim as i32,
        dv: g.gdn_v_dim as i32,
        hk: g.gdn_k_heads as i32,
        hv: g.gdn_v_heads as i32,
        conv_dim: g.gdn_conv_dim as i32,
        kc: g.gdn_conv_k as i32,
        q_off: 0,
        k_off: (g.gdn_k_heads * g.gdn_k_dim) as i32,
        v_off: (2 * g.gdn_k_heads * g.gdn_k_dim) as i32,
        eps: g.eps,
        inv_sqrt_dk: 1.0 / (g.gdn_k_dim as f32).sqrt(),
    }
}

/// `moe/params.h::RouterParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterParams {
    /// The expert count.
    pub n_experts: u32,
    /// Slots per token.
    pub experts_per_token: u32,
    /// 0: softmax the SELECTED logits (weights sum to one;
    /// `norm_topk_prob: true`). 1: softmax over ALL experts then select,
    /// so the weights sum to less than one and scale the routed FFN down.
    pub softmax_over_all: u32,
    /// Elements between logits rows, or 0 for `n_experts`.
    pub logits_pitch: u32,
}
const _: () = assert!(size_of::<RouterParams>() == 16);

/// `moe/params.h::ExpertCombineParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCombineParams {
    /// The output row width.
    pub width: u32,
    /// Slots summed per row.
    pub experts_per_token: u32,
    /// Elements between output rows, or 0 for `width` — nonzero when a
    /// prefill's activations sit a uniform `scratch_widest_elems` apart.
    pub out_pitch: u32,
}
const _: () = assert!(size_of::<ExpertCombineParams>() == 12);

/// `moe/params.h::MoeRouteParams` — ONE struct for the sort and the
/// gather, so the sort's padding and the gather's bounds cannot disagree
/// about how many rows exist.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRouteParams {
    /// The (token, slot) pair count.
    pub n: u32,
    /// The expert count.
    pub n_experts: u32,
    /// Slots per token.
    pub experts_per_token: u32,
    /// The tile each expert's run pads to.
    pub tile_rows: u32,
    /// The padded row count (`moe_sorted_rows`).
    pub padded: u32,
    /// The gathered row width (read only by the gather).
    pub width: u32,
    /// Elements between INPUT rows for the gather, or 0 for `width`.
    pub x_pitch: u32,
}
const _: () = assert!(size_of::<MoeRouteParams>() == 28);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_projection_shapes_are_the_geometrys_and_never_a_defaults() {
        let dense = DecodeGeometry::default();
        assert_eq!(qmv_kn(Kernel::QmvQ, &dense), KN { k: 1024, n: 4096 });
        assert_eq!(
            qmv_kn(Kernel::QmvLmHead, &dense),
            KN {
                k: 1024,
                n: 248_320
            }
        );
        assert_eq!(qmv_kn(Kernel::GdnInA, &dense), KN { k: 1024, n: 16 });
        // The default-geometry defect: a routed kind asked of a dense
        // geometry answers "not a matvec" — which is why the caller must
        // ask its own.
        assert!(!is_qmv(Kernel::ExpertGate, &dense));
        let routed = DecodeGeometry {
            n_experts: 512,
            experts_per_token: 10,
            moe_intermediate: 768,
            ..DecodeGeometry::default()
        };
        assert_eq!(
            qmv_kn(Kernel::ExpertGate, &routed),
            KN { k: 1024, n: 768 }
        );
        assert!(is_routed(Kernel::ExpertDown));
        assert!(!is_routed(Kernel::Router));
    }

    #[test]
    fn the_gdn_params_match_the_kernel_headers_layout() {
        let params = gdn_core_params(&DecodeGeometry::default());
        assert_eq!(params.k_off, 16 * 128);
        assert_eq!(params.v_off, 2 * 16 * 128);
        assert!((params.inv_sqrt_dk - 1.0 / (128.0f32).sqrt()).abs() < 1e-7);
    }
}
