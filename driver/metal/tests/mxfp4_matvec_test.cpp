// The MXFP4 codec, on the GPU, against a reference that shares no code with it.
//
// gpt-oss ships its experts in MXFP4 and everything else -- attention, router,
// embedding, head -- in affine U4. Until now the loader dequantized the experts
// and re-quantized them affine, which is lossy in a way nothing else in the
// checkpoint is, and is the sole reason gpt-oss could not be compared against
// mlx-lm token for token. Reading the published bytes is what closes that.
//
// Three things have to be exactly right and each one fails silently:
//
//   * **The nibble order.** Low nibble first. Swapping them transposes every
//     adjacent pair of weights, which changes the answer by a plausible amount
//     rather than an obvious one.
//   * **The exponent bias.** E8M0 is 127-biased. Off by one is a factor of two
//     on the whole tensor, which a normalizer downstream largely hides.
//   * **The value table.** E2M1 is NOT linear in the code: it steps by 0.5 up
//     to 2 and then by 1, 2, 2. Treating a code as a small integer -- which is
//     exactly what the affine dot does -- is right for the first five entries
//     and wrong for the last three.
//
// So the reference below is written from the format, not from the kernel: a
// table, a bias, and a sum in double. The tolerance is set by the bf16 the
// kernel writes, not by the arithmetic.
//
// Only the matvec: gpt-oss's routed experts never reach the batched matmul.
// Each row picks its own experts, so a tile spanning rows would span weights --
// `gptoss_is_dense_proj` says so and excludes the three -- and the prefill runs
// the same routed matvec with a row axis. An MXFP4 tile loader would be an
// entrypoint nothing dispatches.
//
// K is 2880 because that is gpt-oss's, and because it is the interesting one:
// 2880 = 256 * 11 + 64, so the reduction has a tail that most lanes must sit
// out. A kernel that reads past it reads the next output row's codes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model/gptoss/kernels.hpp"
#include "mtl4_context.hpp"

using pie::metal::Grid;
using pie::metal::Pso;
using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;
using pie::metal::Threadgroup;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

std::uint16_t to_bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    // round-to-nearest-even, as the loader's cast does
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float from_bf16(std::uint16_t v) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float out = 0;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

/// E2M1, written out. Not derived from anything the driver ships.
float mxfp4_value(int code) {
    static const float kTable[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                     -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return kTable[code & 0xf];
}

constexpr int kK = 2880;
constexpr int kN = 16;        // two simdgroups of four results, twice over
constexpr int kExperts = 5;   // more than the slots, so routing has to choose
constexpr int kSlots = 3;

std::uint32_t next(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

}  // namespace

int main() {
    std::printf("[MXFP4 matvec against the format]\n");

    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_FOR_TEST
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
#endif
    if (kernels_dir.empty()) {
        std::printf("  SKIP  no kernels directory\n");
        return 0;
    }
    auto ctx = RawMetalContext::create(std::size_t(1) << 28);
    if (!ctx) {
        std::printf("  SKIP  no Metal device\n");
        return 0;
    }

    std::string err;
    const Pso pso = ctx->compile_pso_from_file(kernels_dir + "/quantized_qmv.metal",
                                               "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", &err);
    if (!pso.valid()) {
        expect(false, "the MXFP4 matvec builds: " + err);
        std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
        return 1;
    }
    expect(true, "the MXFP4 matvec builds");

    // ── the bank ──
    // Codes across the whole 16-entry table, including both zeros and both
    // extremes, so a table that is right in its linear half still fails.
    // Exponents around 127 in both directions, so a sign error on the bias is
    // not cancelled by a symmetric distribution.
    const int rows = kExperts * kN;
    const int words = kK / 8;
    const int groups = kK / 32;
    std::vector<std::uint32_t> w(std::size_t(rows) * words);
    std::vector<std::uint8_t> scales(std::size_t(rows) * groups);
    std::uint32_t seed = 0x243f6a88u;
    for (auto& word : w) word = next(seed);
    for (auto& s : scales) s = static_cast<std::uint8_t>(120 + next(seed) % 15);

    std::vector<std::uint16_t> x(kK);
    for (auto& v : x) v = to_bf16((float(next(seed) >> 8) / float(1u << 24) - 0.5f) * 2.0f);

    std::vector<std::uint16_t> bias(std::size_t(kExperts) * kN);
    for (auto& v : bias) v = to_bf16((float(next(seed) >> 8) / float(1u << 24) - 0.5f) * 0.25f);

    const int chosen[kSlots] = {4, 0, 2};

    SlotHandle w_buf = ctx->create_standalone_buffer(w.size() * sizeof(std::uint32_t));
    SlotHandle s_buf = ctx->create_standalone_buffer(scales.size());
    SlotHandle b_buf = ctx->create_standalone_buffer(bias.size() * 2);
    SlotHandle x_buf = ctx->create_standalone_buffer(x.size() * 2);
    SlotHandle y_buf = ctx->create_standalone_buffer(std::size_t(kSlots) * kN * 2);
    SlotHandle e_buf = ctx->create_standalone_buffer(kSlots * sizeof(int));
    SlotHandle k_buf = ctx->create_standalone_buffer(sizeof(int));
    SlotHandle n_buf = ctx->create_standalone_buffer(sizeof(int));
    SlotHandle xs_buf = ctx->create_standalone_buffer(sizeof(int));
    SlotHandle xr_buf = ctx->create_standalone_buffer(sizeof(int));
    SlotHandle sp_buf = ctx->create_standalone_buffer(sizeof(int));

    std::memcpy(w_buf.contents(), w.data(), w.size() * sizeof(std::uint32_t));
    std::memcpy(s_buf.contents(), scales.data(), scales.size());
    std::memcpy(b_buf.contents(), bias.data(), bias.size() * 2);
    std::memcpy(x_buf.contents(), x.data(), x.size() * 2);
    std::memcpy(e_buf.contents(), chosen, sizeof(chosen));
    // A sentinel the kernel must overwrite: a slot never written reads back as
    // a huge number rather than as a believable zero.
    std::memset(y_buf.contents(), 0x7f, y_buf.size);
    *static_cast<int*>(k_buf.contents()) = kK;
    *static_cast<int*>(n_buf.contents()) = kN;
    *static_cast<int*>(xs_buf.contents()) = 0;  // gate/up read the one shared norm
    *static_cast<int*>(xr_buf.contents()) = kK;
    *static_cast<int*>(sp_buf.contents()) = kSlots;

    const int ordinal = 0;
    ctx->arg_bind_ordinal(ordinal, 0, w_buf);
    ctx->arg_bind_ordinal(ordinal, 1, s_buf);
    ctx->arg_bind_ordinal(ordinal, 2, b_buf);  // unread: MXFP4 has no zero point
    ctx->arg_bind_ordinal(ordinal, 3, x_buf);
    ctx->arg_bind_ordinal(ordinal, 4, y_buf);
    ctx->arg_bind_ordinal(ordinal, 5, k_buf);
    ctx->arg_bind_ordinal(ordinal, 6, n_buf);
    ctx->arg_bind_ordinal(ordinal, 7, b_buf);
    ctx->arg_bind_ordinal(ordinal, 8, e_buf);
    ctx->arg_bind_ordinal(ordinal, 9, xs_buf);
    ctx->arg_bind_ordinal(ordinal, 10, xr_buf);
    ctx->arg_bind_ordinal(ordinal, 11, sp_buf);
    ctx->make_resident();

    Grid grid{};
    Threadgroup tg{};
    pie::metal::gptoss::qmv_dispatch(kN, kSlots, grid, tg, /*rows=*/1);
    ctx->run_step([&](auto& encoder) {
        encoder.set_pso(pso);
        encoder.set_argtable_ordinal(ordinal);
        encoder.dispatch(grid, tg);
    });

    // ── the reference ──
    const auto* got = static_cast<const std::uint16_t*>(y_buf.contents());
    float worst = 0.0f;
    int worst_slot = -1, worst_row = -1;
    for (int slot = 0; slot < kSlots; ++slot) {
        const int e = chosen[slot];
        for (int n = 0; n < kN; ++n) {
            const int row = e * kN + n;
            double acc = 0.0;
            for (int k = 0; k < kK; ++k) {
                const std::uint32_t word = w[std::size_t(row) * words + k / 8];
                const int code = int((word >> (4 * (k % 8))) & 0xf);
                const double factor = std::ldexp(1.0, int(scales[std::size_t(row) * groups + k / 32]) - 127);
                acc += double(from_bf16(x[std::size_t(k)])) * double(mxfp4_value(code)) * factor;
            }
            acc += double(from_bf16(bias[std::size_t(e) * kN + n]));
            const float mine = from_bf16(got[std::size_t(slot) * kN + n]);
            const float denom = std::fmax(1e-6f, float(std::fabs(acc)));
            const float rel = float(std::fabs(mine - float(acc))) / denom;
            if (rel > worst) {
                worst = rel;
                worst_slot = slot;
                worst_row = n;
            }
        }
    }
    char msg[256];
    // bf16 output carries eight mantissa bits, so half an ulp is 2e-3 and a
    // K-long cancellation can lift that. Anything the format could be wrong
    // about moves this by tens of percent, not by a thousandth.
    std::snprintf(msg, sizeof msg,
                  "every routed output matches the format's own arithmetic (worst rel %.5f at "
                  "slot %d row %d)",
                  double(worst), worst_slot, worst_row);
    expect(worst < 6e-3f, msg);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
