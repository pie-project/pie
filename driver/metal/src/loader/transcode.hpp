// transcode.hpp — load-time TileMap execution for the Metal backend.
//
// The loader emits a TileMap when the bytes on disk are not the bytes the
// runtime kernels read. gpt-oss ships its experts as MXFP4 and everything else
// as BF16, while every gpt-oss matvec on Metal reads MLX-style affine U4, so
// the stock checkpoint arrives needing two transforms: decode the MXFP4 blocks
// to BF16, then re-encode a whole tensor to affine U4. Both run ONCE, here, at
// load; after they have run the heap holds exactly the layout an offline
// `mlx_lm convert -q` would have produced, and the decode path is byte-for-byte
// the one it already had.
//
// These are host loops, deliberately. The Metal heap is Shared storage on
// unified memory, so `contents()` is the same memory the GPU reads and a host
// transform needs no staging, no command buffer and no fence — it is a plain
// write to the destination. Threading it across cores is what makes it fast
// enough to be uninteresting; a compute kernel would buy bandwidth this is not
// bound by.
//
// The affine encoder is a transcription of MLX's METAL quantizer,
// `mlx/backend/metal/kernels/quantized.h::affine_quantize`, which the wheel
// ships in source form because it is compiled at runtime. It used to be a
// transcription of the CPU one, and those two are not the same function: on a
// lattice-valued input MLX's own backends disagree on 5.47% of codes, always by
// one, and on one scale in sixteen. mlx-lm quantizes on the GPU here, so the
// GPU one is the only one worth agreeing with.
//
// It is mirrored in `loader/src/testkit/host_executor.rs`, and it has to be: a
// checkpoint converted by MLX and a checkpoint converted here must produce the
// same codes, or the two paths disagree about what the same weights mean.
// `tests/loader_fill_test.cpp` holds both this encoder and the Metal one
// against a pinned `mx.quantize` output, because nothing else does -- no
// checkpoint on disk reaches `push_encoded_affine`.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pie_loader/plan.hpp"

namespace pie::metal::transcode {

namespace lp = pie_loader;

/// Elements per affine group. Fixed by `ScaleLayout::for_encode`, which the
/// contract does not get to choose; asserted rather than read so a change on
/// the Rust side surfaces here instead of producing wrong codes.
inline constexpr std::int64_t kAffineGroup = 64;

/// Run `body(begin, end)` over `[0, count)` split across the machine's cores.
///
/// Ranges rather than indices because every caller's unit of work is a row and
/// the per-row setup (a group's min/max) is worth hoisting.
template <typename Body>
void parallel_ranges(std::int64_t count, const Body& body) {
    if (count <= 0) return;
    const auto hw = static_cast<std::int64_t>(std::thread::hardware_concurrency());
    const std::int64_t workers = std::max<std::int64_t>(1, std::min(count, hw == 0 ? 1 : hw));
    if (workers == 1) {
        body(std::int64_t{0}, count);
        return;
    }
    const std::int64_t chunk = (count + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (std::int64_t w = 0; w < workers; ++w) {
        const std::int64_t begin = w * chunk;
        if (begin >= count) break;
        const std::int64_t end = std::min(count, begin + chunk);
        threads.emplace_back([&body, begin, end] { body(begin, end); });
    }
    for (auto& t : threads) t.join();
}

// ── Element codecs ───────────────────────────────────────────────────────────

inline float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

/// Round-to-nearest-even, matching what a BF16 store does everywhere else in
/// the stack. Truncation would bias every scale downward by half an ulp.
inline std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>(bits >> 16);
    const std::uint32_t rounded =
        bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

/// IEEE-754 binary16 as a float.
///
/// Written out rather than leaned on `_Float16` because this runs on the host
/// for whole tensors and the whole point is that the bit layout is EXPLICIT:
/// F16 and BF16 differ in where the exponent ends, which is precisely the
/// confusion that reading one as the other produces.
inline float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    std::uint32_t wide = 0;
    if (exp == 0x1fu) {
        wide = sign | 0x7f800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant != 0) {
            // Subnormal: renormalise into binary32, which has the range for it.
            std::uint32_t m = mant;
            std::int32_t e = -1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            wide = sign | (static_cast<std::uint32_t>(e + 127 - 14) << 23) | (m << 13);
        } else {
            wide = sign;
        }
    } else {
        wide = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

/// One MXFP4 nibble as a float. E2M1: sign, two exponent bits, one mantissa
/// bit, with the all-zero exponent denormal at ±0.5.
inline float mxfp4_nibble(std::uint8_t code) {
    static constexpr float kLut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return kLut[code & 0xf];
}

/// An E8M0 block exponent as a float multiplier. 0xff is the NaN encoding.
inline float e8m0_scale(std::uint8_t code) {
    if (code == 0xff) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

// ── MLX affine quantization ──────────────────────────────────────────────────

/// The (scale, bias) for one group, at the precision they are STORED in.
///
/// Two details of the endpoints are load-bearing and neither is guessable:
///   * the scale is NEGATED unless the negative extreme is the larger in
///     magnitude, so scales are usually negative;
///   * the endpoint, not the scale, is snapped -- `scale = edge / round(edge /
///     scale)` -- which keeps the largest-magnitude weight exactly representable
///     and is why a naive `(max - min) / 15` disagrees with MLX on real data;
///   * `w_max` starts at ZERO, not at negative infinity, so a group whose
///     values are all negative is quantized over the range up to zero rather
///     than up to its own largest element. Nothing about the arithmetic
///     suggests this; it is simply what MLX does.
///
/// The rounding is `round`, half AWAY FROM ZERO, and not `rint`, half to even.
/// That one character was the whole of a long-standing 8.2% disagreement with
/// `mx.quantize`. A tie at k+0.5 goes up under MLX and down-to-even here, which
/// is why every mismatch was by exactly one and almost always in the same
/// direction -- and why the rate was 0.2% on gaussian weights and 8.2% on an
/// MXFP4-derived expert bank, whose values sit on half-integers by
/// construction. It was previously read as evidence that agreement with MLX was
/// not purchasable at all, and the encoder was changed to pick codes against
/// the BF16-rounded scale instead. It is purchasable, and it is exact: with
/// `round`, zero-initialised `w_max`, and the f32 parameters, this reproduces
/// `mx.quantize` bit for bit on every distribution tried.
///
/// The cost of going back to the f32 parameters is real and tiny. MLX's own
/// codes are not the best ones for the BF16 scale MLX itself stores -- 0.23% of
/// them could be moved and reduce the error -- but the reconstruction RMSE
/// differs in the fifth significant digit, and mlx-lm redeems those same codes
/// at that same scale. Matching the thing being compared against beats being
/// 0.001% better than it.
struct AffineGroup {
    float scale = 0.0f;
    float bias = 0.0f;
};

inline AffineGroup mlx_affine_group_params(const float* values, std::int64_t count) {
    float w_min = std::numeric_limits<float>::infinity();
    float w_max = 0.0f;
    for (std::int64_t i = 0; i < count; ++i) {
        w_min = std::min(w_min, values[i]);
        w_max = std::max(w_max, values[i]);
    }
    const bool mask = std::abs(w_min) > std::abs(w_max);
    float scale = std::max((w_max - w_min) / 15.0f, 1e-7f);
    if (!mask) scale = -scale;
    const float edge = mask ? w_min : w_max;
    const float q0 = std::round(edge / scale);
    float bias = 0.0f;
    if (q0 != 0.0f) {
        scale = edge / q0;
        bias = edge;
    }
    // At f32, the width MLX picks codes at. They are written as BF16 by the
    // caller, which is also what MLX does -- it rounds only on store.
    return AffineGroup{scale, bias};
}

// ── Buffer plumbing ──────────────────────────────────────────────────────────

/// A destination the executor can write: raw bytes plus their extent.
struct Region {
    std::uint8_t* data = nullptr;
    std::uint64_t bytes = 0;
};

inline void require(bool condition, const char* what) {
    if (!condition) {
        throw std::runtime_error(std::string("metal transcode: ") + what);
    }
}

/// Rewrite `count` F16 or F32 elements as BF16.
///
/// The narrowing every checkpoint that is not already BF16 needs, and the one
/// that cannot be skipped: `mlx-community` publishes some conversions in F16
/// and some in BF16, the two are the same width, and a driver that transmuted
/// rather than cast would read one as the other without a single byte moving.
/// The result is not noise -- it is a model that loads, runs and says the same
/// token forever.
///
/// Widths are 2 and 4 rather than a dtype enum because this file has no
/// dependency on the loader's ABI and does not want one; the caller has the
/// dtype and this wants only how wide it is.
inline void cast_float_to_bf16(const std::uint8_t* src, std::uint32_t src_bytes,
                               std::int64_t count, Region dst) {
    require(src_bytes == 2 || src_bytes == 4, "a float cast reads 2 or 4 bytes per element");
    require(dst.bytes >= static_cast<std::uint64_t>(count) * 2,
            "a float cast has no room for its output");
    auto* out = reinterpret_cast<std::uint16_t*>(dst.data);
    parallel_ranges(count, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            float value;
            if (src_bytes == 2) {
                std::uint16_t bits;
                std::memcpy(&bits, src + i * 2, 2);
                value = f16_to_f32(bits);
            } else {
                std::memcpy(&value, src + i * 4, 4);
            }
            out[i] = f32_to_bf16(value);
        }
    });
}

/// Multiply `payload` by per-block factors, writing BF16.
///
/// This is the MXFP4 decode: `transform_from` says the payload's nibbles are
/// E2M1, `factors` holds one E8M0 byte per block, and the block size falls out
/// of the two shapes. Emitted by `scale_per_block` in a contract.
inline void scale_mxfp4_to_bf16(
    const std::uint8_t* payload,
    std::uint64_t payload_bytes,
    const std::uint8_t* factors,
    std::uint64_t factor_count,
    Region out) {
    const std::uint64_t elements = payload_bytes * 2;
    require(out.bytes == elements * 2, "MXFP4 decode output is not BF16-shaped");
    require(factor_count != 0, "MXFP4 decode has no block exponents");
    require(elements % factor_count == 0, "MXFP4 block size does not divide the payload");
    const std::uint64_t block = elements / factor_count;
    require(block % 2 == 0, "MXFP4 block does not land on a byte boundary");

    auto* dst = reinterpret_cast<std::uint16_t*>(out.data);
    parallel_ranges(static_cast<std::int64_t>(factor_count), [&](std::int64_t begin,
                                                                std::int64_t end) {
        for (std::int64_t b = begin; b < end; ++b) {
            const float factor = e8m0_scale(factors[b]);
            const std::uint64_t first = static_cast<std::uint64_t>(b) * block;
            for (std::uint64_t i = 0; i < block; i += 2) {
                const std::uint8_t byte = payload[(first + i) / 2];
                dst[first + i] = f32_to_bf16(mxfp4_nibble(byte & 0xf) * factor);
                dst[first + i + 1] = f32_to_bf16(mxfp4_nibble(byte >> 4) * factor);
            }
        }
    });
}

/// Quantize `rows x cols` BF16 (or F32) to MLX affine U4: codes packed eight per
/// little-endian u32, plus BF16 scales and biases of `[rows, cols / 64]`.
inline void encode_mlx_affine_u4(
    const std::uint8_t* input,
    std::uint32_t input_element_bytes,
    std::int64_t rows,
    std::int64_t cols,
    Region codes,
    Region scales,
    Region biases) {
    require(cols % kAffineGroup == 0, "affine encode needs a multiple of 64 columns");
    const std::int64_t groups = cols / kAffineGroup;
    require(codes.bytes == static_cast<std::uint64_t>(rows * cols) / 2,
            "affine encode code buffer is the wrong size");
    require(scales.bytes == static_cast<std::uint64_t>(rows * groups) * 2 &&
                biases.bytes == scales.bytes,
            "affine encode metadata buffer is the wrong size");
    require(input_element_bytes == 2 || input_element_bytes == 4,
            "affine encode operand is neither BF16 nor F32");

    auto* code_words = reinterpret_cast<std::uint32_t*>(codes.data);
    auto* scale_out = reinterpret_cast<std::uint16_t*>(scales.data);
    auto* bias_out = reinterpret_cast<std::uint16_t*>(biases.data);

    parallel_ranges(rows, [&](std::int64_t begin, std::int64_t end) {
        std::vector<float> group(static_cast<std::size_t>(kAffineGroup));
        for (std::int64_t r = begin; r < end; ++r) {
            for (std::int64_t g = 0; g < groups; ++g) {
                const std::int64_t first = r * cols + g * kAffineGroup;
                for (std::int64_t i = 0; i < kAffineGroup; ++i) {
                    const std::uint8_t* at =
                        input + static_cast<std::uint64_t>(first + i) * input_element_bytes;
                    if (input_element_bytes == 2) {
                        std::uint16_t bits;
                        std::memcpy(&bits, at, sizeof(bits));
                        group[static_cast<std::size_t>(i)] = bf16_to_f32(bits);
                    } else {
                        float value;
                        std::memcpy(&value, at, sizeof(value));
                        group[static_cast<std::size_t>(i)] = value;
                    }
                }
                const AffineGroup params = mlx_affine_group_params(group.data(), kAffineGroup);
                scale_out[r * groups + g] = f32_to_bf16(params.scale);
                bias_out[r * groups + g] = f32_to_bf16(params.bias);
                // Against the f32 parameters, not the BF16 ones written above:
                // that is what MLX does, and being interchangeable with an
                // `mlx_lm convert` output is the entire purpose of this file.
                for (std::int64_t w = 0; w < kAffineGroup / 8; ++w) {
                    std::uint32_t packed = 0;
                    for (std::int64_t k = 0; k < 8; ++k) {
                        const float value = group[static_cast<std::size_t>(w * 8 + k)];
                        const float q = std::round((value - params.bias) / params.scale);
                        const auto code = static_cast<std::uint32_t>(
                            std::min(15.0f, std::max(0.0f, q)));
                        packed |= code << (4 * k);
                    }
                    code_words[(r * cols + g * kAffineGroup) / 8 + w] = packed;
                }
            }
        }
    });
}

}  // namespace pie::metal::transcode
