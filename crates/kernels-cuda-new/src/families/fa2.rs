use crate::unit::Unit;

/// `kernels.def`'s `PIE_ATTN_HEAD_DIM` list, in its order.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `kernels.def`'s `PIE_ATTN_DECODE_GQA` list, in its order.
pub const DECODE_GQA: &[u32] = &[1, 2, 3, 4, 8];

/// `PagedTraits` and the six variant aliases live in one header, and every unit
const ROOT: &str = include_str!("../../csrc/src/attn/fa2.cuh");

/// `--device-as-default-execution-space`, and it is load-bearing.
const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

/// One decode unit: five rows over one `(head_dim, GQA group)` point.
macro_rules! decode_unit {
    (
        $unit:ident, hd = $hd:literal, gqa = $g:literal,
        stages = $ns:literal, tile = $tile:literal, vec = $vec:literal,
        bdx = $bdx:literal, bdy = $bdy:literal, bdz = $bdz:literal,
        $(#[$note:meta])*
    ) => {
        $(#[$note])*
        pub mod $unit {
            use kernels::{KernelSig, kernel};

            use super::{OPTIONS, ROOT};
            use crate::device::DeviceKernel;
            use crate::unit::Unit;

            /// The head dim and GQA group this unit is the lattice point for.
            pub const POINT: (u32, u32) = ($hd, $g);

            const PATH: &str = "::flashinfer::BatchDecodeWithPagedKVCacheKernel";

            #[rustfmt::skip]
            static SIGS: [KernelSig; 5] = [
                kernel!(decode_full
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_full"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_softcap
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_softcap"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_window
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_window"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_capture_full
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_capture_full"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_capture_window
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_capture_window"),
                    file = Some("attn/fa2.cuh")),
            ];

            #[rustfmt::skip]
            static ROWS: [DeviceKernel; 5] = [
                DeviceKernel { sig: &SIGS[0], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[1], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[2], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindow, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[3], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CaptureFull, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams") },
                DeviceKernel { sig: &SIGS[4], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CaptureWindow, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams") },
            ];

            /// The unit itself.
            pub const UNIT: Unit = Unit {
                name: concat!(
                    "attn/fa2_decode_hd", stringify!($hd), "_g", stringify!($g)
                ),
                root: ROOT,
                rows: &ROWS,
                options: OPTIONS,
            };
        }
    };
}

/// One prefill unit: ten rows over one `(head_dim, CTA_TILE_Q, NUM_MMA_KV)`
macro_rules! prefill_unit {
    (
        $unit:ident, hd = $hd:literal, q = $q:literal, kv = $kv:literal,
        mma_q = $mmaq:literal, d_qk = $dqk:literal, d_vo = $dvo:literal,
        warps_q = $wq:literal, warps_kv = $wkv:literal,
        $(#[$note:meta])*
    ) => {
        $(#[$note])*
        pub mod $unit {
            use kernels::{KernelSig, kernel};

            use super::{OPTIONS, ROOT};
            use crate::device::DeviceKernel;
            use crate::unit::Unit;

            /// The head dim, `CTA_TILE_Q` and `NUM_MMA_KV` this unit is the
            pub const POINT: (u32, u32, u32) = ($hd, $q, $kv);

            const PATH: &str = "::flashinfer::BatchPrefillWithPagedKVCacheKernel";

            #[rustfmt::skip]
            static SIGS: [KernelSig; 10] = [
                kernel!(prefill_causal_full_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_full_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_full_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_full_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_full concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_full"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_full concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_full"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_window concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_window"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_capture concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_capture"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_capture concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_capture"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_custom_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_custom_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_custom concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_custom"), file = Some("attn/fa2.cuh")),
            ];

            #[rustfmt::skip]
            static ROWS: [DeviceKernel; 10] = [
                DeviceKernel { sig: &SIGS[0], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[1], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[2], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[3], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[4], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[5], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindow>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[6], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams") },
                DeviceKernel { sig: &SIGS[7], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams") },
                DeviceKernel { sig: &SIGS[8], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantCustomSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[9], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantCustom>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
            ];

            /// The unit itself.
            pub const UNIT: Unit = Unit {
                name: concat!(
                    "attn/fa2_prefill_hd", stringify!($hd),
                    "_q", stringify!($q), "_kv", stringify!($kv)
                ),
                root: ROOT,
                rows: &ROWS,
                options: OPTIONS,
            };
        }
    };
}

decode_unit!(d_hd64_g1, hd = 64, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 8, bdy = 1, bdz = 16,
    /// llama3.2-1B, MQA. `tile_size_per_bdx = 4` is the GQA-1 special case
);
decode_unit!(d_hd64_g2, hd = 64, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 2, bdz = 8,);
decode_unit!(d_hd64_g3, hd = 64, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 3, bdz = 5,
    /// `128 / (8*3) = 5`, so the block is 8x3x5 = **120 threads**. Upstream
);
decode_unit!(d_hd64_g4, hd = 64, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 4, bdz = 4,);
decode_unit!(d_hd64_g8, hd = 64, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 8, bdz = 2,);

decode_unit!(d_hd128_g1, hd = 128, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 16, bdy = 1, bdz = 8,);
decode_unit!(d_hd128_g2, hd = 128, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 2, bdz = 4,);
decode_unit!(d_hd128_g3, hd = 128, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 3, bdz = 2,);
decode_unit!(d_hd128_g4, hd = 128, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 4, bdz = 2,
    /// qwen3 and qwen2's usual shape, and **the point the pre-port NVRTC probe
);
decode_unit!(d_hd128_g8, hd = 128, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 8, bdz = 1,);

decode_unit!(d_hd256_g1, hd = 256, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 32, bdy = 1, bdz = 4,);
decode_unit!(d_hd256_g2, hd = 256, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 2, bdz = 2,);
decode_unit!(d_hd256_g3, hd = 256, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 3, bdz = 1,);
decode_unit!(d_hd256_g4, hd = 256, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 4, bdz = 1,);
decode_unit!(d_hd256_g8, hd = 256, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 8, bdz = 1,
    /// The one decode point where `num_threads` exceeds 128: `bdx*bdy = 256`,
);

decode_unit!(d_hd512_g1, hd = 512, gqa = 1, stages = 2, tile = 4, vec = 16, bdx = 32, bdy = 1, bdz = 4,
    /// **69,632 B of dynamic shared memory** — over the 48 KB default cap, so
);
decode_unit!(d_hd512_g2, hd = 512, gqa = 2, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 2, bdz = 2,);
decode_unit!(d_hd512_g3, hd = 512, gqa = 3, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 3, bdz = 1,);
decode_unit!(d_hd512_g4, hd = 512, gqa = 4, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 4, bdz = 1,);
decode_unit!(d_hd512_g8, hd = 512, gqa = 8, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 8, bdz = 1,);

prefill_unit!(p_hd64_q16_kv8, hd = 64, q = 16, kv = 8, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd64_q16_kv4, hd = 64, q = 16, kv = 4, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd64_q16_kv2, hd = 64, q = 16, kv = 2, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,
    /// `NUM_MMA_KV = 1` is absent at head dim 64 for every tile:
);
prefill_unit!(p_hd64_q64_kv8, hd = 64, q = 64, kv = 8, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q64_kv4, hd = 64, q = 64, kv = 4, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q64_kv2, hd = 64, q = 64, kv = 2, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv8, hd = 64, q = 128, kv = 8, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv4, hd = 64, q = 128, kv = 4, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv2, hd = 64, q = 128, kv = 2, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);

prefill_unit!(p_hd128_q16_kv8, hd = 128, q = 16, kv = 8, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv4, hd = 128, q = 16, kv = 4, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv2, hd = 128, q = 16, kv = 2, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv1, hd = 128, q = 16, kv = 1, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q64_kv8, hd = 128, q = 64, kv = 8, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv4, hd = 128, q = 64, kv = 4, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv2, hd = 128, q = 64, kv = 2, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv1, hd = 128, q = 64, kv = 1, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q128_kv4, hd = 128, q = 128, kv = 4, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,
    /// `NUM_MMA_KV = 8` is absent here: `2 * (8*8 + 8*8) = 256`, which is
);
prefill_unit!(p_hd128_q128_kv2, hd = 128, q = 128, kv = 2, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q128_kv1, hd = 128, q = 128, kv = 1, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);

prefill_unit!(p_hd256_q16_kv8, hd = 256, q = 16, kv = 8, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv4, hd = 256, q = 16, kv = 4, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv2, hd = 256, q = 16, kv = 2, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv1, hd = 256, q = 16, kv = 1, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q64_kv8, hd = 256, q = 64, kv = 8, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv4, hd = 256, q = 64, kv = 4, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv2, hd = 256, q = 64, kv = 2, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv1, hd = 256, q = 64, kv = 1, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,
    /// **`CTA_TILE_Q = 128` has no valid point at head dim 256.**
);

prefill_unit!(p_hd512_q16_kv8, hd = 512, q = 16, kv = 8, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv4, hd = 512, q = 16, kv = 4, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv2, hd = 512, q = 16, kv = 2, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv1, hd = 512, q = 16, kv = 1, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q32_kv8, hd = 512, q = 32, kv = 8, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,
    /// `kBf16VOSplit` (`prefill.cuh:4191`): 16-bit KV, head dim >= 512 and
);
prefill_unit!(p_hd512_q32_kv4, hd = 512, q = 32, kv = 4, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);
prefill_unit!(p_hd512_q32_kv2, hd = 512, q = 32, kv = 2, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);
prefill_unit!(p_hd512_q32_kv1, hd = 512, q = 32, kv = 1, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);

/// Every FA2 unit: 20 decode, 36 prefill.
pub const UNITS: &[Unit] = &[
    d_hd64_g1::UNIT,
    d_hd64_g2::UNIT,
    d_hd64_g3::UNIT,
    d_hd64_g4::UNIT,
    d_hd64_g8::UNIT,
    d_hd128_g1::UNIT,
    d_hd128_g2::UNIT,
    d_hd128_g3::UNIT,
    d_hd128_g4::UNIT,
    d_hd128_g8::UNIT,
    d_hd256_g1::UNIT,
    d_hd256_g2::UNIT,
    d_hd256_g3::UNIT,
    d_hd256_g4::UNIT,
    d_hd256_g8::UNIT,
    d_hd512_g1::UNIT,
    d_hd512_g2::UNIT,
    d_hd512_g3::UNIT,
    d_hd512_g4::UNIT,
    d_hd512_g8::UNIT,
    p_hd64_q16_kv8::UNIT,
    p_hd64_q16_kv4::UNIT,
    p_hd64_q16_kv2::UNIT,
    p_hd64_q64_kv8::UNIT,
    p_hd64_q64_kv4::UNIT,
    p_hd64_q64_kv2::UNIT,
    p_hd64_q128_kv8::UNIT,
    p_hd64_q128_kv4::UNIT,
    p_hd64_q128_kv2::UNIT,
    p_hd128_q16_kv8::UNIT,
    p_hd128_q16_kv4::UNIT,
    p_hd128_q16_kv2::UNIT,
    p_hd128_q16_kv1::UNIT,
    p_hd128_q64_kv8::UNIT,
    p_hd128_q64_kv4::UNIT,
    p_hd128_q64_kv2::UNIT,
    p_hd128_q64_kv1::UNIT,
    p_hd128_q128_kv4::UNIT,
    p_hd128_q128_kv2::UNIT,
    p_hd128_q128_kv1::UNIT,
    p_hd256_q16_kv8::UNIT,
    p_hd256_q16_kv4::UNIT,
    p_hd256_q16_kv2::UNIT,
    p_hd256_q16_kv1::UNIT,
    p_hd256_q64_kv8::UNIT,
    p_hd256_q64_kv4::UNIT,
    p_hd256_q64_kv2::UNIT,
    p_hd256_q64_kv1::UNIT,
    p_hd512_q16_kv8::UNIT,
    p_hd512_q16_kv4::UNIT,
    p_hd512_q16_kv2::UNIT,
    p_hd512_q16_kv1::UNIT,
    p_hd512_q32_kv8::UNIT,
    p_hd512_q32_kv4::UNIT,
    p_hd512_q32_kv2::UNIT,
    p_hd512_q32_kv1::UNIT,
];

/// The unit that holds one decode lattice point, by name.
#[must_use]
pub fn decode_unit_name(head_dim: u32, group_size: u32) -> Option<&'static str> {
    let unit = match (head_dim, group_size) {
        (64, 1) => d_hd64_g1::UNIT,
        (64, 2) => d_hd64_g2::UNIT,
        (64, 3) => d_hd64_g3::UNIT,
        (64, 4) => d_hd64_g4::UNIT,
        (64, 8) => d_hd64_g8::UNIT,
        (128, 1) => d_hd128_g1::UNIT,
        (128, 2) => d_hd128_g2::UNIT,
        (128, 3) => d_hd128_g3::UNIT,
        (128, 4) => d_hd128_g4::UNIT,
        (128, 8) => d_hd128_g8::UNIT,
        (256, 1) => d_hd256_g1::UNIT,
        (256, 2) => d_hd256_g2::UNIT,
        (256, 3) => d_hd256_g3::UNIT,
        (256, 4) => d_hd256_g4::UNIT,
        (256, 8) => d_hd256_g8::UNIT,
        (512, 1) => d_hd512_g1::UNIT,
        (512, 2) => d_hd512_g2::UNIT,
        (512, 3) => d_hd512_g3::UNIT,
        (512, 4) => d_hd512_g4::UNIT,
        (512, 8) => d_hd512_g8::UNIT,
        _ => return None,
    };
    Some(unit.name)
}

/// One decode row's symbol, by lattice point and arm.
#[must_use]
pub fn decode_symbol(head_dim: u32, group_size: u32, arm: DecodeArm) -> Option<&'static str> {
    let name = decode_unit_name(head_dim, group_size)?;
    let unit = UNITS.iter().find(|unit| unit.name == name)?;
    unit.rows.get(arm as usize).map(|row| row.sig.symbol)
}

/// Which of `dispatch_decode`'s five branches a fire took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    /// `AttnVariantFull` — full attention, no window, no soft cap.
    Full = 0,
    /// `AttnVariantSoftcap` — a soft cap, windowed or not.
    Softcap = 1,
    /// `AttnVariant` — the sliding-window default.
    Window = 2,
    /// `AttnScoreCaptureFull` over `DecodeScoreParams`.
    CaptureFull = 3,
    /// `AttnScoreCapture` over `DecodeScoreParams`.
    CaptureWindow = 4,
}

/// The unit that holds one prefill lattice point, by name.
#[must_use]
pub fn prefill_unit_name(head_dim: u32, cta_tile_q: u32, num_mma_kv: u32) -> Option<&'static str> {
    let unit = match (head_dim, cta_tile_q, num_mma_kv) {
        (64, 16, 8) => p_hd64_q16_kv8::UNIT,
        (64, 16, 4) => p_hd64_q16_kv4::UNIT,
        (64, 16, 2) => p_hd64_q16_kv2::UNIT,
        (64, 64, 8) => p_hd64_q64_kv8::UNIT,
        (64, 64, 4) => p_hd64_q64_kv4::UNIT,
        (64, 64, 2) => p_hd64_q64_kv2::UNIT,
        (64, 128, 8) => p_hd64_q128_kv8::UNIT,
        (64, 128, 4) => p_hd64_q128_kv4::UNIT,
        (64, 128, 2) => p_hd64_q128_kv2::UNIT,
        (128, 16, 8) => p_hd128_q16_kv8::UNIT,
        (128, 16, 4) => p_hd128_q16_kv4::UNIT,
        (128, 16, 2) => p_hd128_q16_kv2::UNIT,
        (128, 16, 1) => p_hd128_q16_kv1::UNIT,
        (128, 64, 8) => p_hd128_q64_kv8::UNIT,
        (128, 64, 4) => p_hd128_q64_kv4::UNIT,
        (128, 64, 2) => p_hd128_q64_kv2::UNIT,
        (128, 64, 1) => p_hd128_q64_kv1::UNIT,
        (128, 128, 4) => p_hd128_q128_kv4::UNIT,
        (128, 128, 2) => p_hd128_q128_kv2::UNIT,
        (128, 128, 1) => p_hd128_q128_kv1::UNIT,
        (256, 16, 8) => p_hd256_q16_kv8::UNIT,
        (256, 16, 4) => p_hd256_q16_kv4::UNIT,
        (256, 16, 2) => p_hd256_q16_kv2::UNIT,
        (256, 16, 1) => p_hd256_q16_kv1::UNIT,
        (256, 64, 8) => p_hd256_q64_kv8::UNIT,
        (256, 64, 4) => p_hd256_q64_kv4::UNIT,
        (256, 64, 2) => p_hd256_q64_kv2::UNIT,
        (256, 64, 1) => p_hd256_q64_kv1::UNIT,
        (512, 16, 8) => p_hd512_q16_kv8::UNIT,
        (512, 16, 4) => p_hd512_q16_kv4::UNIT,
        (512, 16, 2) => p_hd512_q16_kv2::UNIT,
        (512, 16, 1) => p_hd512_q16_kv1::UNIT,
        (512, 32, 8) => p_hd512_q32_kv8::UNIT,
        (512, 32, 4) => p_hd512_q32_kv4::UNIT,
        (512, 32, 2) => p_hd512_q32_kv2::UNIT,
        (512, 32, 1) => p_hd512_q32_kv1::UNIT,
        _ => return None,
    };
    Some(unit.name)
}

/// One prefill row's symbol, by lattice point and arm.
#[must_use]
pub fn prefill_symbol(
    head_dim: u32,
    cta_tile_q: u32,
    num_mma_kv: u32,
    arm: PrefillArm,
) -> Option<&'static str> {
    let name = prefill_unit_name(head_dim, cta_tile_q, num_mma_kv)?;
    let unit = UNITS.iter().find(|unit| unit.name == name)?;
    unit.rows.get(arm as usize).map(|row| row.sig.symbol)
}

/// Which of the ten prefill branches a fire took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillArm {
    /// `kCausal`, `AttnVariantFullSoftcap`.
    CausalFullSoftcap = 0,
    /// `kNone`, `AttnVariantFullSoftcap`.
    NoneFullSoftcap = 1,
    /// `kCausal`, `AttnVariantFull`.
    CausalFull = 2,
    /// `kNone`, `AttnVariantFull`.
    NoneFull = 3,
    /// `kCausal`, `AttnVariantSoftcap` — the windowed soft-cap variant, and
    CausalSoftcap = 4,
    /// `kCausal`, `AttnVariant`.
    CausalWindow = 5,
    /// `kCausal`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    CausalCapture = 6,
    /// `kNone`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    NoneCapture = 7,
    /// `kCustom`, `AttnVariantCustomSoftcap`.
    CustomSoftcap = 8,
    /// `kCustom`, `AttnVariantCustom`.
    Custom = 9,
}

#[cfg(test)]
mod tests {
    use super::{DECODE_GQA, HEAD_DIMS, UNITS, decode_unit_name, prefill_unit_name};
    use crate::fa2::{Device, KvWidth, PrefillGeometry};

    /// Every decode row's six template constants are the ones
    #[test]
    fn decode_literals_match_the_derivation() {
        for &head_dim in HEAD_DIMS {
            for &group in DECODE_GQA {
                let name = decode_unit_name(head_dim, group)
                    .unwrap_or_else(|| panic!("no unit for hd {head_dim} gqa {group}"));
                let unit = UNITS.iter().find(|unit| unit.name == name).unwrap();
                let geometry = crate::fa2::DecodeGeometry::derive(
                    head_dim,
                    group,
                    KvWidth::BF16,
                    Device::L40S,
                )
                .unwrap_or_else(|why| panic!("hd {head_dim} gqa {group}: {why}"));
                let wanted = format!(
                    "::flashinfer::PosEncodingMode::kNone, {}, {}, {}, {}, {}, {}, ",
                    geometry.num_stages_smem,
                    geometry.tile_size_per_bdx,
                    geometry.vec_size,
                    geometry.bdx,
                    geometry.bdy,
                    geometry.bdz,
                );
                for row in unit.rows {
                    assert!(
                        row.elem.starts_with(&wanted),
                        "hd {head_dim} gqa {group}: row states\n  {}\nderivation wants\n  {wanted}",
                        row.elem,
                    );
                }
            }
        }
    }

    /// Every prefill row's `KernelTraits` arguments are the ones
    #[test]
    fn prefill_literals_match_the_derivation() {
        for unit in UNITS.iter().filter(|unit| unit.name.contains("fa2_prefill")) {
            let point = unit
                .name
                .trim_start_matches("attn/fa2_prefill_hd")
                .split(['_'])
                .collect::<Vec<_>>();
            let head_dim: u32 = point[0].parse().unwrap();
            let cta_tile_q: u32 = point[1].trim_start_matches('q').parse().unwrap();
            let num_mma_kv: u32 = point[2].trim_start_matches("kv").parse().unwrap();
            let geometry =
                PrefillGeometry::derive(head_dim, cta_tile_q, KvWidth::BF16, true, Device::L40S)
                    .unwrap_or_else(|why| panic!("{}: {why}", unit.name));
            let wanted = format!(
                ", {cta_tile_q}, {}, {num_mma_kv}, {}, {}, {}, {}, ",
                geometry.num_mma_q,
                geometry.num_mma_d_qk,
                geometry.num_mma_d_vo,
                geometry.num_warps_q,
                geometry.num_warps_kv,
            );
            for row in unit.rows {
                assert!(
                    row.elem.contains(&wanted),
                    "{}: row states\n  {}\nderivation wants\n  {wanted}",
                    unit.name,
                    row.elem,
                );
            }
        }
    }

    /// The `NUM_MMA_KV` the derivation picks on this box names a unit that
    #[test]
    fn the_derived_num_mma_kv_names_a_unit() {
        for &head_dim in HEAD_DIMS {
            for &cta_tile_q in &[16u32, 32, 64, 128] {
                let Ok(geometry) = PrefillGeometry::derive(
                    head_dim,
                    cta_tile_q,
                    KvWidth::BF16,
                    true,
                    Device::L40S,
                ) else {
                    continue;
                };
                assert!(
                    prefill_unit_name(head_dim, cta_tile_q, geometry.num_mma_kv).is_some(),
                    "hd {head_dim} q {cta_tile_q} derived NUM_MMA_KV {} and no unit holds it",
                    geometry.num_mma_kv,
                );
            }
        }
    }

    /// No two FA2 units share a name and no two rows share a symbol.
    #[test]
    fn names_and_symbols_are_unique() {
        let mut names: Vec<&str> = Vec::new();
        let mut symbols: Vec<&str> = Vec::new();
        for unit in UNITS {
            assert!(!names.contains(&unit.name), "{} is declared twice", unit.name);
            names.push(unit.name);
            for row in unit.rows {
                assert!(
                    !symbols.contains(&row.sig.symbol),
                    "{} is stated twice",
                    row.sig.symbol
                );
                symbols.push(row.sig.symbol);
            }
        }
        assert_eq!(UNITS.len(), 56);
        assert_eq!(symbols.len(), 20 * 5 + 36 * 10);
    }

    /// Every FA2 row spells an ABSOLUTE instantiation.
    #[test]
    fn every_row_is_absolutely_qualified() {
        for unit in UNITS {
            for row in unit.rows {
                let name = row.instantiation();
                assert!(
                    name.starts_with("::flashinfer::"),
                    "{} instantiates {name}",
                    row.sig.symbol
                );
                assert!(
                    !name.contains("::pie_cuda_driver::kernels::::"),
                    "{} double-qualified: {name}",
                    row.sig.symbol
                );
            }
        }
    }
}
