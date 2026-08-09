// `boost/math/ccmath/fabs.hpp` for NVRTC -- a constexpr absolute value, under the name
// FlashInfer asks for.
//
// `flashinfer/fp16.h` computes `boost::math::ccmath::fabs<float>(f) * scale_to_inf` in a
// `constexpr` function that the prefill kernel calls on the device. Boost is not on this
// machine, is not in the FlashInfer tarball, and would be 400 headers if it were; what is
// actually wanted is one branchless line. `std::fabs` is not constexpr before C++23 and
// `::fabsf` is a device intrinsic NVRTC supplies but not a constant expression, which is
// why upstream reached for Boost's `ccmath` in the first place.
//
// So this file impersonates the name and implements the line. It contains no Boost code
// and is not derived from Boost: it is a two-line ternary written from the definition of
// absolute value. It is PIE's file. The name is Boost's only because an `#include`
// resolves against a header set by exact spelling and nothing else -- this crate has no
// include path on disk, so the spelling in the directive IS the identity of the header.
//
// If FlashInfer ever calls another `ccmath` function, the compile says so by name and the
// fix is another file beside this one.

#ifndef PIE_VENDOR_BOOST_MATH_CCMATH_FABS_HPP_
#define PIE_VENDOR_BOOST_MATH_CCMATH_FABS_HPP_

namespace boost {
namespace math {
namespace ccmath {

/// The absolute value of `x`, evaluated at compile time when its argument is constant --
/// the contract `boost::math::ccmath::fabs` offers and the one `fp16.h` relies on.
template <typename T>
constexpr T fabs(T x) noexcept {
  return x < T(0) ? -x : x;
}

}  // namespace ccmath
}  // namespace math
}  // namespace boost

#endif  // PIE_VENDOR_BOOST_MATH_CCMATH_FABS_HPP_
