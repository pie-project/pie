#pragma once

#include <cstdint>

// One input of a column concat: its rows' `width` bytes into `out`'s rows
// at byte `at` of each (`stride` bytes a row), 16 bytes a thread where the
// widths allow, else a byte.
namespace pie::layout {

__global__ void concat_cols(const unsigned char* __restrict__ x, unsigned char* __restrict__ out,
                            int rows, int width, int stride, int at, int wide) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wide) {
        const int per = width / 16;
        if (i >= static_cast<long long>(rows) * per) return;
        const int r = static_cast<int>(i / per), c = static_cast<int>(i % per) * 16;
        *reinterpret_cast<uint4*>(out + static_cast<long long>(r) * stride + at + c) =
            *reinterpret_cast<const uint4*>(x + static_cast<long long>(r) * width + c);
    } else {
        if (i >= static_cast<long long>(rows) * width) return;
        const int r = static_cast<int>(i / width), c = static_cast<int>(i % width);
        out[static_cast<long long>(r) * stride + at + c] = x[static_cast<long long>(r) * width + c];
    }
}

}  // namespace pie::layout
