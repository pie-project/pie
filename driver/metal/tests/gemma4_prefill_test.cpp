// Does gemma4's attention mean the same thing at M>1 as it does at M=1?
//
// A prefill is only correct if row m of an M-row launch produces exactly what a
// single-token step would have produced for that token. That is the invariant
// teacher-forced agreement rests on, and it is cheap to check directly: run the
// M-row launch once, then run M single-row launches with the key count each row
// should see, and compare.
//
// Checked on-device, because the thing being tested is a kernel.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "batch/decode_abi.hpp"
#include "model/gemma4/kernels.hpp"
#include "kernels/decode_psos.hpp"
#include "model/qwen3_5/decode_dispatch_mb.hpp"

using namespace pie::metal;
using namespace pie::metal::gemma4;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// bf16 round-trip, matching what the driver stages into activation buffers.
std::uint16_t to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t(rounded >> 16);
}
float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Deterministic, so a failure is reproducible.
float noise(int i) {
    std::uint32_t x = std::uint32_t(i) * 2654435761u + 12345u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return float(int(x % 2001) - 1000) / 1000.0f;
}

constexpr int kD = 256;          // sliding head_dim
constexpr int kHeads = 8;        // q heads
constexpr int kKvHeads = 1;
constexpr int kMaxCtx = 1024;

struct Fixture {
    RawMetalContext& ctx;
    SlotHandle q, k, v, out;

    // One argument-table ordinal per launch shape, so the two launches cannot
    // contaminate each other's binds.
    void bind(int ord, int n_keys, int window, int q_rows_stride, int o_rows_stride,
              std::size_t q_off_elems, std::size_t o_off_elems) {
        using B = bind::SdpaSliding;
        ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Q, q, q_off_elems * 2);
        ctx.arg_bind_ordinal(ord, (std::uint8_t)B::K, k);
        ctx.arg_bind_ordinal(ord, (std::uint8_t)B::V, v);
        ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Out, out, o_off_elems * 2);
        auto i32 = [&](B idx, std::int32_t val) {
            SlotHandle s = ctx.heap_alloc(4);
            *static_cast<std::int32_t*>(s.contents()) = val;
            ctx.arg_bind_ordinal(ord, (std::uint8_t)idx, s);
        };
        auto sz = [&](B idx, std::uint64_t val) {
            SlotHandle s = ctx.heap_alloc(8);
            *static_cast<std::uint64_t*>(s.contents()) = val;
            ctx.arg_bind_ordinal(ord, (std::uint8_t)idx, s);
        };
        auto f32 = [&](B idx, float val) {
            SlotHandle s = ctx.heap_alloc(4);
            *static_cast<float*>(s.contents()) = val;
            ctx.arg_bind_ordinal(ord, (std::uint8_t)idx, s);
        };
        i32(B::GqaFactor, kHeads / kKvHeads);
        i32(B::N, n_keys);
        sz(B::KHeadStride, std::uint64_t(kMaxCtx) * kD);
        sz(B::KSeqStride, kD);
        sz(B::VHeadStride, std::uint64_t(kMaxCtx) * kD);
        sz(B::VSeqStride, kD);
        f32(B::Scale, 1.0f / std::sqrt(float(kD)));
        i32(B::Window, window);
        i32(B::QRowStride, q_rows_stride);
        i32(B::ORowStride, o_rows_stride);
    }
};

void a_prefill_row_matches_the_decode_step_that_would_have_made_it(
    RawMetalContext& ctx, const Gemma4Psos& psos, int window, const char* label) {
    std::printf("[prefill row == decode step, window=%s]\n", label);

    const int M = 6;              // rows in the prefill launch
    const int n_keys = 700;       // > sliding window (512), so the window bites

    Fixture f{ctx, {}, {}, {}, {}};
    f.q = ctx.heap_alloc(std::size_t(M) * kHeads * kD * 2);
    f.k = ctx.heap_alloc(std::size_t(kKvHeads) * kMaxCtx * kD * 2);
    f.v = ctx.heap_alloc(std::size_t(kKvHeads) * kMaxCtx * kD * 2);
    f.out = ctx.heap_alloc(std::size_t(M) * kHeads * kD * 2 * 2);  // both halves
    if (!f.q.valid() || !f.k.valid() || !f.v.valid() || !f.out.valid()) {
        expect(false, "fixture allocates");
        return;
    }

    auto* qp = static_cast<std::uint16_t*>(f.q.contents());
    for (int i = 0; i < M * kHeads * kD; ++i) qp[i] = to_bf16(noise(i) * 0.5f);
    auto* kp = static_cast<std::uint16_t*>(f.k.contents());
    auto* vp = static_cast<std::uint16_t*>(f.v.contents());
    for (int i = 0; i < kKvHeads * kMaxCtx * kD; ++i) {
        kp[i] = to_bf16(noise(i + 7777) * 0.5f);
        vp[i] = to_bf16(noise(i + 4242) * 0.5f);
    }
    std::memset(f.out.contents(), 0, f.out.size);

    const int q_stride = kHeads * kD;
    const std::size_t decode_half = std::size_t(M) * kHeads * kD;  // second half of `out`

    // Bind every ordinal, then make the heap resident once. Binding after
    // residency leaves the GPU unable to see the slots, which writes nothing at
    // all rather than writing something wrong.
    f.bind(900, n_keys, window, q_stride, q_stride, 0, 0);
    for (int m = 0; m < M; ++m) {
        f.bind(901 + m, n_keys - (M - 1 - m), window, q_stride, q_stride,
               std::size_t(m) * q_stride, decode_half + std::size_t(m) * q_stride);
    }
    ctx.make_resident();

    // One launch of M rows.
    ctx.run_step([&](StepEncoder& se) {
        se.set_pso(psos.sdpa_swa_d256);
        se.set_argtable_ordinal(900);
        Grid grid; Threadgroup tg;
        sdpa_sliding_dispatch(kHeads, grid, tg, M);
        se.dispatch(grid, tg);
        se.barrier();
    });

    // M launches of one row. Row m's token is the (M-1-m)-th from the end, so it
    // sees n_keys-(M-1-m) keys -- which is exactly what it saw as a decode step.
    for (int m = 0; m < M; ++m) {
        ctx.run_step([&](StepEncoder& se) {
            se.set_pso(psos.sdpa_swa_d256);
            se.set_argtable_ordinal(901 + m);
            Grid grid; Threadgroup tg;
            sdpa_sliding_dispatch(kHeads, grid, tg, 1);
            se.dispatch(grid, tg);
            se.barrier();
        });
    }

    const auto* op = static_cast<const std::uint16_t*>(f.out.contents());
    // Did anything get written at all? An all-zero buffer makes every
    // comparison below pass vacuously, which is the failure mode of a
    // comparison test that nobody checked for signal.
    int nz_prefill = 0, nz_decode = 0;
    for (int i = 0; i < M * kHeads * kD; ++i) {
        nz_prefill += op[i] != 0 ? 1 : 0;
        nz_decode += op[decode_half + i] != 0 ? 1 : 0;
    }
    std::printf("    (nonzero: prefill %d/%d, decode %d/%d)\n", nz_prefill,
                M * kHeads * kD, nz_decode, M * kHeads * kD);
    expect(nz_prefill > 0 && nz_decode > 0, "both launches actually wrote output");
    int mismatched_rows = 0;
    float worst = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < kHeads * kD; ++i) {
            const float a = from_bf16(op[std::size_t(m) * q_stride + i]);
            const float b = from_bf16(op[decode_half + std::size_t(m) * q_stride + i]);
            const float d = std::fabs(a - b);
            if (d > worst) worst = d;
        }
        bool row_ok = true;
        for (int i = 0; i < kHeads * kD && row_ok; ++i) {
            row_ok = op[std::size_t(m) * q_stride + i] ==
                     op[decode_half + std::size_t(m) * q_stride + i];
        }
        if (!row_ok) ++mismatched_rows;
    }
    std::printf("    (M=%d rows, %d keys, worst abs delta %.3e)\n", M, n_keys, worst);
    expect(mismatched_rows == 0,
           "every prefill row is bit-identical to its decode step (" +
               std::to_string(M - mismatched_rows) + "/" + std::to_string(M) + ")");

    // The rows must not all be the same answer -- if they were, the comparison
    // above would pass while the causal position did nothing.
    bool rows_differ = false;
    for (int i = 0; i < kHeads * kD && !rows_differ; ++i) {
        rows_differ = op[i] != op[std::size_t(M - 1) * q_stride + i];
    }
    expect(rows_differ, "different rows attend different spans (the position is live)");
}

// A pure elementwise kernel over a contiguous [M, N] buffer is already an M>1
// kernel: pass n = M*N. Asserted rather than assumed, because "it's elementwise"
// is exactly the kind of claim that is true until someone adds a row-dependent
// term to it.
void the_elementwise_kernels_are_already_batched(RawMetalContext& ctx,
                                                 const Gemma4Psos& psos) {
    std::printf("[elementwise kernels widen by n = M*width]\n");
    const int M = 5, N = 384;
    const int total = M * N;

    SlotHandle a = ctx.heap_alloc(std::size_t(total) * 2);
    SlotHandle b = ctx.heap_alloc(std::size_t(total) * 2);
    SlotHandle o = ctx.heap_alloc(std::size_t(total) * 2 * 2);
    if (!a.valid() || !b.valid() || !o.valid()) {
        expect(false, "fixture allocates");
        return;
    }
    auto* ap = static_cast<std::uint16_t*>(a.contents());
    auto* bp = static_cast<std::uint16_t*>(b.contents());
    for (int i = 0; i < total; ++i) {
        ap[i] = to_bf16(noise(i + 31) * 2.0f);
        bp[i] = to_bf16(noise(i + 91) * 2.0f);
    }
    std::memset(o.contents(), 0, o.size);

    auto run_geglu = [&](int ord, std::size_t in_off, std::size_t out_off, int n,
                         std::uint32_t grid_n) {
        ctx.run_step([&](StepEncoder& se) {
            using B = bind::Geglu;
            ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Gate, a, in_off * 2);
            ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Up, b, in_off * 2);
            ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Out, o, out_off * 2);
            SlotHandle pr = ctx.heap_alloc(4);
            *static_cast<std::uint32_t*>(pr.contents()) = std::uint32_t(n);
            ctx.arg_bind_ordinal(ord, (std::uint8_t)B::Params, pr);
            se.set_pso(psos.geglu_tanh);
            se.set_argtable_ordinal(ord);
            se.dispatch(Grid{grid_n, 1, 1}, Threadgroup{256, 1, 1});
            se.barrier();
        });
    };

    ctx.make_resident();
    // One launch over all M rows...
    run_geglu(800, 0, 0, total, std::uint32_t((total + 255) / 256 * 256));
    // ...against M launches of one row each.
    for (int m = 0; m < M; ++m) {
        run_geglu(801 + m, std::size_t(m) * N, std::size_t(total) + std::size_t(m) * N, N,
                  std::uint32_t((N + 255) / 256 * 256));
    }

    const auto* op = static_cast<const std::uint16_t*>(o.contents());
    int bad = 0;
    for (int i = 0; i < total; ++i) bad += op[i] != op[total + i] ? 1 : 0;
    expect(bad == 0, "geglu_tanh over M*N equals M launches over N");
}

// The paged kernel and the contiguous one must agree.
//
// Two independent implementations of sliding attention is exactly the situation
// that produces a model which works at decode and quietly does something else at
// prefill. With one kv head and one page, the paged layout
// [pages, page_size, kv_heads, D] and the contiguous [kv_heads, max_ctx, D]
// coincide, so the same bytes can be fed to both and the outputs compared.
void the_paged_kernel_agrees_with_the_contiguous_one(RawMetalContext& ctx,
                                                     const Gemma4Psos& psos,
                                                     const DecodeStepPsos& base,
                                                     const MultiBatchPsos& mb,
                                                     int window, const char* label) {
    std::printf("[paged == contiguous, window=%s]\n", label);
    const int M = 4;
    const int n_keys = 600;

    SlotHandle q = ctx.heap_alloc(std::size_t(M) * kHeads * kD * 2);
    SlotHandle k = ctx.heap_alloc(std::size_t(kMaxCtx) * kD * 2);
    SlotHandle v = ctx.heap_alloc(std::size_t(kMaxCtx) * kD * 2);
    SlotHandle out = ctx.heap_alloc(std::size_t(M) * kHeads * kD * 2 * 2);
    SlotHandle pos = ctx.heap_alloc(std::size_t(M) * 4);
    SlotHandle req = ctx.heap_alloc(std::size_t(M) * 4);
    SlotHandle pidx = ctx.heap_alloc(4);
    SlotHandle pptr = ctx.heap_alloc(8);
    SlotHandle mask_on = ctx.heap_alloc(std::size_t(M));
    SlotHandle mask = ctx.heap_alloc(std::size_t(M));
    if (!q.valid() || !k.valid() || !out.valid() || !pptr.valid()) {
        expect(false, "fixture allocates");
        return;
    }
    auto* qp = static_cast<std::uint16_t*>(q.contents());
    for (int i = 0; i < M * kHeads * kD; ++i) qp[i] = to_bf16(noise(i + 5) * 0.5f);
    auto* kp = static_cast<std::uint16_t*>(k.contents());
    auto* vp = static_cast<std::uint16_t*>(v.contents());
    for (int i = 0; i < kMaxCtx * kD; ++i) {
        kp[i] = to_bf16(noise(i + 7777) * 0.5f);
        vp[i] = to_bf16(noise(i + 4242) * 0.5f);
    }
    std::memset(out.contents(), 0, out.size);
    // Row m holds the token at position n_keys-M+m, matching the contiguous
    // kernel's own N-(M-1-m).
    for (int m = 0; m < M; ++m) {
        static_cast<std::int32_t*>(pos.contents())[m] = n_keys - M + m;
        static_cast<std::int32_t*>(req.contents())[m] = 0;
    }
    static_cast<std::uint32_t*>(pidx.contents())[0] = 0;
    static_cast<std::uint32_t*>(pptr.contents())[0] = 0;
    static_cast<std::uint32_t*>(pptr.contents())[1] = 1;
    std::memset(mask_on.contents(), 0, mask_on.size);  // no mask
    std::memset(mask.contents(), 1, mask.size);

    const int q_stride = kHeads * kD;
    const std::size_t second = std::size_t(M) * kHeads * kD;

    auto i32 = [&](int ord, std::uint8_t idx, std::int32_t val) {
        SlotHandle sl = ctx.heap_alloc(4);
        *static_cast<std::int32_t*>(sl.contents()) = val;
        ctx.arg_bind_ordinal(ord, idx, sl);
    };
    auto f32 = [&](int ord, std::uint8_t idx, float val) {
        SlotHandle sl = ctx.heap_alloc(4);
        *static_cast<float*>(sl.contents()) = val;
        ctx.arg_bind_ordinal(ord, idx, sl);
    };
    auto u64 = [&](int ord, std::uint8_t idx, std::uint64_t val) {
        SlotHandle sl = ctx.heap_alloc(8);
        *static_cast<std::uint64_t*>(sl.contents()) = val;
        ctx.arg_bind_ordinal(ord, idx, sl);
    };

    // Paged, into the first half of `out`.
    {
        using B = bind::SdpaPaged;
        const int o = 700;
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::Q, q);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::KPages, k);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::VPages, v);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::Out, out);
        i32(o, (std::uint8_t)B::GqaFactor, kHeads / kKvHeads);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::PositionIds, pos);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::ReqOfToken, req);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::KvPageIndices, pidx);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::KvPageIndptr, pptr);
        i32(o, (std::uint8_t)B::PageSize, kMaxCtx);
        i32(o, (std::uint8_t)B::NKvHeads, kKvHeads);
        f32(o, (std::uint8_t)B::Scale, 1.0f / std::sqrt(float(kD)));
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::AttnMask, mask);
        u64(o, (std::uint8_t)B::AttnMaskStride, 1);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::AttnMaskEnabled, mask_on);
        i32(o, (std::uint8_t)B::Window, window);
    }
    // Contiguous, into the second half.
    {
        using B = bind::SdpaSliding;
        const int o = 701;
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::Q, q);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::K, k);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::V, v);
        ctx.arg_bind_ordinal(o, (std::uint8_t)B::Out, out, second * 2);
        i32(o, (std::uint8_t)B::GqaFactor, kHeads / kKvHeads);
        i32(o, (std::uint8_t)B::N, n_keys);
        u64(o, (std::uint8_t)B::KHeadStride, std::uint64_t(kMaxCtx) * kD);
        u64(o, (std::uint8_t)B::KSeqStride, kD);
        u64(o, (std::uint8_t)B::VHeadStride, std::uint64_t(kMaxCtx) * kD);
        u64(o, (std::uint8_t)B::VSeqStride, kD);
        f32(o, (std::uint8_t)B::Scale, 1.0f / std::sqrt(float(kD)));
        i32(o, (std::uint8_t)B::Window, window);
        i32(o, (std::uint8_t)B::QRowStride, q_stride);
        i32(o, (std::uint8_t)B::ORowStride, q_stride);
    }
    ctx.make_resident();

    ctx.run_step([&](StepEncoder& se) {
        Grid g; Threadgroup t;
        sdpa_paged_dispatch(kHeads, M, g, t);
        se.set_pso(mb.sdpa_paged);
        se.set_argtable_ordinal(700);
        se.dispatch(g, t);
        se.barrier();
    });
    ctx.run_step([&](StepEncoder& se) {
        Grid g; Threadgroup t;
        sdpa_sliding_dispatch(kHeads, g, t, M);
        se.set_pso(psos.sdpa_swa_d256);
        se.set_argtable_ordinal(701);
        se.dispatch(g, t);
        se.barrier();
    });

    const auto* op = static_cast<const std::uint16_t*>(out.contents());
    int nz = 0, bad = 0;
    float worst = 0.0f;
    for (int i = 0; i < M * kHeads * kD; ++i) {
        nz += op[i] != 0 ? 1 : 0;
        const float a = from_bf16(op[i]), b = from_bf16(op[second + i]);
        worst = std::fabs(a - b) > worst ? std::fabs(a - b) : worst;
        bad += op[i] != op[second + i] ? 1 : 0;
    }
    std::printf("    (nonzero %d/%d, mismatched %d, worst %.3e)\n", nz,
                M * kHeads * kD, bad, worst);
    expect(nz > 0, "the paged kernel wrote output");
    expect(bad == 0, "paged sliding == contiguous sliding, element for element");
    (void)base;
}

// Exactly `window` positions, including the query's own.
//
// vLLM, mlx-lm and llama.cpp all settled on `q - k < window` (strict), so a
// window of W covers [q-W+1, q] -- W keys counting self. An off-by-one here is
// the single most-cited pitfall in all three, and it is invisible in output:
// attending W-1 or W+1 keys produces a plausible number.
//
// Pinned at the boundary rather than in the middle, where a fencepost hides: a
// window of exactly N must equal full attention, and N-1 must not.
void the_window_covers_exactly_that_many_positions_counting_self(
    RawMetalContext& ctx, const Gemma4Psos& psos) {
    std::printf("[window boundary: W keys including self]\n");
    const int n_keys = 64;

    SlotHandle q = ctx.heap_alloc(std::size_t(kHeads) * kD * 2);
    SlotHandle k = ctx.heap_alloc(std::size_t(kMaxCtx) * kD * 2);
    SlotHandle v = ctx.heap_alloc(std::size_t(kMaxCtx) * kD * 2);
    SlotHandle out = ctx.heap_alloc(std::size_t(kHeads) * kD * 2 * 3);
    if (!q.valid() || !out.valid()) {
        expect(false, "fixture allocates");
        return;
    }
    auto* qp = static_cast<std::uint16_t*>(q.contents());
    for (int i = 0; i < kHeads * kD; ++i) qp[i] = to_bf16(noise(i + 13) * 0.5f);
    auto* kp = static_cast<std::uint16_t*>(k.contents());
    auto* vp = static_cast<std::uint16_t*>(v.contents());
    for (int i = 0; i < kMaxCtx * kD; ++i) {
        kp[i] = to_bf16(noise(i + 7777) * 0.5f);
        vp[i] = to_bf16(noise(i + 4242) * 0.5f);
    }
    std::memset(out.contents(), 0, out.size);

    const int stride = kHeads * kD;
    Fixture f{ctx, q, k, v, out};
    // arm 0: full attention. arm 1: window == n_keys. arm 2: window == n_keys-1.
    f.bind(600, n_keys, 0, stride, stride, 0, 0);
    f.bind(601, n_keys, n_keys, stride, stride, 0, std::size_t(stride));
    f.bind(602, n_keys, n_keys - 1, stride, stride, 0, std::size_t(2) * stride);
    ctx.make_resident();
    for (int arm = 0; arm < 3; ++arm) {
        ctx.run_step([&](StepEncoder& se) {
            Grid g; Threadgroup t;
            sdpa_sliding_dispatch(kHeads, g, t, 1);
            se.set_pso(psos.sdpa_swa_d256);
            se.set_argtable_ordinal(600 + arm);
            se.dispatch(g, t);
            se.barrier();
        });
    }

    const auto* op = static_cast<const std::uint16_t*>(out.contents());
    int full_vs_exact = 0, full_vs_short = 0;
    for (int i = 0; i < stride; ++i) {
        full_vs_exact += op[i] != op[stride + i] ? 1 : 0;
        full_vs_short += op[i] != op[2 * stride + i] ? 1 : 0;
    }
    expect(full_vs_exact == 0,
           "a window of exactly N is full attention (so N keys, counting self)");
    expect(full_vs_short > 0,
           "a window of N-1 is not (so the boundary is where it should be)");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
    if (argc > 1) kernels_dir = argv[1];
    std::printf("gemma4 prefill kernels (%s)\n", kernels_dir.c_str());

    auto ctx = RawMetalContext::create(64u << 20);
    if (!ctx) {
        std::printf("  FAIL  RawMetalContext::create\n");
        return 1;
    }
    Gemma4Psos psos;
    std::string error;
    if (!build_gemma4_psos(*ctx, kernels_dir, Gemma4Geometry{}, psos, &error)) {
        std::printf("  FAIL  build_gemma4_psos: %s\n", error.c_str());
        return 1;
    }

    a_prefill_row_matches_the_decode_step_that_would_have_made_it(*ctx, psos, 512, "512");
    a_prefill_row_matches_the_decode_step_that_would_have_made_it(*ctx, psos, 0, "0 (full)");
    the_window_covers_exactly_that_many_positions_counting_self(*ctx, psos);
    the_elementwise_kernels_are_already_batched(*ctx, psos);

    DecodeStepPsos base;
    MultiBatchPsos mb;
    std::string berr;
    // The gemma4 checkpoints these tests read are affine b4/g64; this test is
    // about pipeline resolution, not the quantization axis.
    const AffineFormat q{/*bits=*/4, /*group=*/64};
    if (load_decode_psos(
            *ctx, kernels_dir, base, q, &berr,
            DecodePsoFeatures{.argmax = true}) &&
        load_multibatch_psos(
            *ctx, kernels_dir, mb, q, &berr,
            MultiBatchPsoFeatures{.d512 = true, .sdpa_d256 = true})) {
        the_paged_kernel_agrees_with_the_contiguous_one(*ctx, psos, base, mb, 512, "512");
        the_paged_kernel_agrees_with_the_contiguous_one(*ctx, psos, base, mb, 0, "0 (full)");
    } else {
        expect(false, "the shared PSOs compile: " + berr);
    }

    std::printf("\n==== gemma4_prefill_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
