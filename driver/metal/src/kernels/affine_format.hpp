#pragma once

#include <string>

namespace pie::metal {

/// The two numbers that together name every quantized entrypoint in this
/// driver: the affine width and the affine group. They are one fact, not two.
///
/// Neither is inferable from the tensors. g64/b8 and g128/b4 pack to
/// identical shapes, so the checkpoint's `config.json` is the only source,
/// and a pipeline built for the wrong pair does not fail -- it reads the
/// scales against the wrong weights and returns fluent nonsense. (Observed:
/// a g64 pipeline over a g32 checkpoint answers token 3504, repeated.)
///
/// They live in one struct because passing one without the other has been an
/// actual defect twice. When they were adjacent defaulted parameters, call
/// sites passed the width and let the group fall back to 64, which compiled,
/// bound, dispatched, and lied. A pair that travels as a single value cannot
/// be half-supplied, and the suffix is spelled here once instead of at every
/// place that builds a name.
struct AffineFormat {
    int bits;
    int group;

    /// The trailing segment shared by every quantized kernel name:
    /// `<base>_bfloat16_gs_<group>_b_<bits>`. bf16 is not an axis -- it is the
    /// activation dtype the whole driver uses, and the only one instantiated.
    std::string kernel_suffix() const {
        return "_bfloat16_gs_" + std::to_string(group) + "_b_" + std::to_string(bits);
    }
};

}  // namespace pie::metal
