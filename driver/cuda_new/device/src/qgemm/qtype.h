#pragma once
//
// Minimal compile-time scalar-type tag for the fused quant GEMM.
//
// This replaces the external torch-coupled ScalarType class the upstream
// kernel used as a non-type template parameter. We keep the same essential
// shape so the lifted kernel/dequant/dtype templates compile with minimal
// edits:
//   * a small literal-class `QType` carrying (exponent, mantissa, signed_, bias)
//   * a stable compile-time `id()` (packed bitfield) usable as a non-type
//     template argument under C++17, plus `from_id()`
//   * `size_bits()` and value-equality
//   * the named constants the kernel body references in `if constexpr` /
//     `static_assert` branches.
//
// Only the bf16 x u4b8 path is *instantiated* in this slice; the other
// constants (fp16, fp8, fp4, int8, ...) exist purely so the template's
// compile-time-discarded branches type-check. Adding a real fp8/mxfp4/int8
// path later only requires new dequant specializations + instantiations, not
// changes here.

#include <cstdint>
#include <tuple>
#include <utility>

#if defined(__CUDACC__)
  #define QGEMM_HD __host__ __device__
#else
  #define QGEMM_HD
#endif

namespace pie_cuda_device::qgemm {

class QType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,
    NAN_IEEE_754 = 1,
    NAN_EXTD_RANGE_MAX_MIN = 2,
    NAN_REPR_ID_MAX
  };

  QGEMM_HD constexpr QType(uint8_t exponent, uint8_t mantissa, bool signed_,
                  int32_t bias, bool finite_values_only = false,
                  NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {}

  QGEMM_HD static constexpr QType int_(uint8_t size_bits, int32_t bias = 0) {
    return QType(0, size_bits - 1, true, bias);
  }
  QGEMM_HD static constexpr QType uint(uint8_t size_bits, int32_t bias = 0) {
    return QType(0, size_bits, false, bias);
  }
  QGEMM_HD static constexpr QType float_IEEE754(uint8_t exponent, uint8_t mantissa) {
    return QType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }
  QGEMM_HD static constexpr QType float_(uint8_t exponent, uint8_t mantissa,
                                bool finite_values_only, NanRepr nan_repr) {
    return QType(exponent, mantissa, true, 0, finite_values_only, nan_repr);
  }

  uint8_t const exponent;
  uint8_t const mantissa;
  bool const signed_;
  int32_t const bias;
  bool const finite_values_only;
  NanRepr const nan_repr;

  using Id = int64_t;

 private:
  // Field bit-widths, in the order the id packs them (matches the original
  // construction order so the values are stable across the codebase):
  //   exponent(8) mantissa(8) signed_(1) bias(32) finite_values_only(1)
  //   nan_repr(8)  => 58 bits, fits in an int64.
  static constexpr int kBExp = 8, kBMant = 8, kBSign = 1, kBBias = 32,
                       kBFinite = 1, kBNan = 8;
  QGEMM_HD static constexpr Id mask_bits(int bits) {
    return (Id(1) << bits) - 1;
  }

 public:
  // Stable compile-time id, usable as a non-type template parameter under
  // C++17 (literal classes can't be passed directly until C++20). Computed
  // with plain integer ops so it is valid in device code without pulling
  // std::tuple/std::pair into __device__ context.
  QGEMM_HD constexpr Id id() const {
    Id v = 0;
    int off = 0;
    v |= (Id(exponent) & mask_bits(kBExp)) << off;            off += kBExp;
    v |= (Id(mantissa) & mask_bits(kBMant)) << off;           off += kBMant;
    v |= (Id(signed_ ? 1 : 0) & mask_bits(kBSign)) << off;    off += kBSign;
    v |= (Id(bias) & mask_bits(kBBias)) << off;               off += kBBias;
    v |= (Id(finite_values_only ? 1 : 0) & mask_bits(kBFinite)) << off;
    off += kBFinite;
    v |= (Id(nan_repr) & mask_bits(kBNan)) << off;
    return v;
  }

  QGEMM_HD static constexpr QType from_id(Id id) {
    int off = 0;
    uint8_t exp = static_cast<uint8_t>((id >> off) & mask_bits(kBExp));
    off += kBExp;
    uint8_t mant = static_cast<uint8_t>((id >> off) & mask_bits(kBMant));
    off += kBMant;
    bool sgn = static_cast<bool>((id >> off) & mask_bits(kBSign));
    off += kBSign;
    int32_t bs = static_cast<int32_t>((id >> off) & mask_bits(kBBias));
    off += kBBias;
    bool fin = static_cast<bool>((id >> off) & mask_bits(kBFinite));
    off += kBFinite;
    NanRepr nr = static_cast<NanRepr>((id >> off) & mask_bits(kBNan));
    return QType(exp, mant, sgn, bs, fin, nr);
  }

  QGEMM_HD constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }
  QGEMM_HD constexpr bool is_signed() const { return signed_; }
  QGEMM_HD constexpr bool is_integer() const { return exponent == 0; }
  QGEMM_HD constexpr bool is_floating_point() const { return exponent > 0; }
  QGEMM_HD constexpr bool has_bias() const { return bias != 0; }

  QGEMM_HD constexpr bool operator==(QType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using QTypeId = QType::Id;

// Integer weight/scale tags.
static inline constexpr auto kS4 = QType::int_(4);
static inline constexpr auto kU4 = QType::uint(4);
static inline constexpr auto kU4B8 = QType::uint(4, 8);  // GPTQ symmetric int4
static inline constexpr auto kS8 = QType::int_(8);
static inline constexpr auto kU8 = QType::uint(8);
static inline constexpr auto kU8B128 = QType::uint(8, 128);

// Floating-point tags (present so compile-time-discarded branches type-check).
static inline constexpr auto kFE2M1f =
    QType::float_(2, 1, true, QType::NAN_NONE);
static inline constexpr auto kFE3M2f =
    QType::float_(3, 2, true, QType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    QType::float_(4, 3, true, QType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    QType(8, 0, false, 0, true, QType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = QType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = QType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = QType::float_IEEE754(5, 10);

static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

}  // namespace pie_cuda_device::qgemm
