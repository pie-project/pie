#pragma once

/// A dispatch's shape, and nothing else.
///
/// These live on the kernels' side because a launch shape is a property of the
/// kernel, not of the model that dispatches it — the same sentence
/// `driver-metal`'s `shared_kernels.hpp` opens with about buffer layouts, one
/// crate further down. A threadgroup width that must be a whole simdgroup is
/// that because the SHADER reduces across simdgroups; nothing about the caller
/// says so.
///
/// Two PODs and no behaviour, so this header pulls in no Metal, no
/// Objective-C, and nothing from the driver. That is what lets
/// `include/pie/kernels/*.h` state launch geometry at all: the alternative is
/// for the shapes to stay upstairs, where they cannot see the kernel they
/// describe.
///
/// `driver-metal` aliases these into `pie::metal` rather than redeclaring
/// them, so `Grid{...}` reads the same at every call site it always did.

#include <cstdint>

namespace pie::kernels {

struct Grid {
    std::uint32_t x = 1, y = 1, z = 1;
};

struct Threadgroup {
    std::uint32_t x = 1, y = 1, z = 1;
};

}  // namespace pie::kernels
