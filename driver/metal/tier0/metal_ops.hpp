#pragma once
//
// MetalOps — dispatches PTIR tier-0 compute ops to the validated Metal kernels
// (driver/metal/ptir/kernels/sampling_ir.metal). Pure C++ over MetalHarness
// (the ObjC++ boundary), so the tier-0 runner stays host C++.

#include <cstdint>
#include <string>
#include <vector>

#include "metal_harness.hpp"

namespace tier0 {

using ptir_metal::Arg;
using ptir_metal::MetalHarness;

struct MetalOps {
    MetalHarness& h;
    bool ok = true;

    bool load(const std::string& kernels_dir) {
        return h.load_library(kernels_dir + "/sampling_ir.metal");
    }

    // Per-row argmax over [rows, len] f32 -> I32[rows] (ReduceArgmax).
    std::vector<std::int32_t> reduce_argmax(const std::vector<float>& in, int rows, int len) {
        std::vector<std::int32_t> out(rows, 0);
        int r = rows, l = len;
        std::vector<Arg> a = {Arg::in(in.data(), in.size() * 4), Arg::out(out.data(), rows * 4),
                              Arg::in(&r, 4), Arg::in(&l, 4)};
        ok = ok && h.run("reduce_argmax_rows", a, rows);
        return out;
    }
    // mask_apply_packed vector: out[j] = bit ? logits[j] : -inf.
    std::vector<float> mask_apply_packed(const std::vector<float>& logits,
                                         const std::vector<std::uint32_t>& mask) {
        std::vector<float> out(logits.size(), 0.0f);
        std::uint32_t n = (std::uint32_t)logits.size();
        std::vector<Arg> a = {Arg::in(logits.data(), logits.size() * 4),
                              Arg::in(mask.data(), mask.size() * 4),
                              Arg::out(out.data(), out.size() * 4), Arg::in(&n, 4)};
        ok = ok && h.run("mask_apply_packed", a, n);
        return out;
    }
    // dselect: out[i] = cond ? a : b (f32), full-length operands.
    std::vector<float> dselect(const std::vector<std::uint8_t>& cond, const std::vector<float>& a,
                               const std::vector<float>& b) {
        std::uint32_t n = (std::uint32_t)cond.size(), lc = n, la = (std::uint32_t)a.size(),
                      lb = (std::uint32_t)b.size();
        std::vector<float> out(n, 0.0f);
        std::vector<Arg> ar = {Arg::in(cond.data(), n), Arg::in(a.data(), a.size() * 4),
                               Arg::in(b.data(), b.size() * 4), Arg::out(out.data(), out.size() * 4),
                               Arg::in(&n, 4), Arg::in(&lc, 4), Arg::in(&la, 4), Arg::in(&lb, 4)};
        ok = ok && h.run("dselect_f32", ar, n);
        return out;
    }
    // Per-row reduce sum/max/min over [rows,len].
    std::vector<float> reduce_rows(const char* fn, const std::vector<float>& in, int rows, int len) {
        std::vector<float> out(rows, 0.0f);
        int r = rows, l = len;
        std::vector<Arg> a = {Arg::in(in.data(), in.size() * 4), Arg::out(out.data(), rows * 4),
                              Arg::in(&r, 4), Arg::in(&l, 4)};
        ok = ok && h.run(fn, a, rows);
        return out;
    }
};

}  // namespace tier0
