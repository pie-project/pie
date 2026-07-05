// Beam [B,P] kvm packed-mask adapter (SEAM 1) unit test — the fork-freeze golden
// is the adapter's correctness gate (echo: "the 3 fork-freeze/page-turn/continue
// goldens ARE the adapter's test vectors"). Verifies launch_beam_pack_kvm packs
// the dense per-cell kvm over each beam's klen span into FlashInfer's bit-packed
// bitmap, bit-exact against a host oracle, on the fork-freeze geometry.
//
// Self-contained: no test framework; failures abort non-zero for CTest.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/pack_dense_mask.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                \
                         __FILE__, __LINE__, msg);                   \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

#define CUDA_RT(call)                                                \
    do {                                                             \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_e));          \
            std::exit(2);                                            \
        }                                                            \
    } while (0)

// Host oracle: bit `j` of beam b = kvm_dense[b][j] for j in [0, klen[b]).
std::vector<std::uint8_t> pack_oracle(
    const std::vector<std::uint8_t>& kvm, const std::vector<std::uint32_t>& klen,
    const std::vector<std::int32_t>& indptr, int B, int P_PAGE) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(indptr[B]), 0);
    for (int b = 0; b < B; ++b) {
        const int kl = static_cast<int>(klen[b]);
        std::uint8_t* row = out.data() + indptr[b];
        for (int j = 0; j < kl; ++j)
            if (kvm[static_cast<std::size_t>(b) * P_PAGE + j])
                row[j / 8] |= static_cast<std::uint8_t>(1u << (j % 8));
    }
    return out;
}

void test_fork_freeze() {
    // Golden geometry (interface/sampling-ir/tests/ptir_examples.rs:232): P=3,
    // PAGE=4, BB=2. Pages=[5,6,7|5,6,0], klen=[9,7]. kvm per-cell over [B,P*PAGE=12]:
    //  lane 0 (frozen): page0(slot5) full, page1(slot6) off0,1 valid, off2 HOLE
    //   (the heir's appended token — freeze), off3 unfilled; page2(slot7) off0 valid.
    //  lane 1 (heir): page0 full, page1 off0,1,2 valid (sees own append), rest.
    const int B = 2, P = 3, PAGE = 4;
    const int P_PAGE = P * PAGE;  // 12
    std::vector<std::uint32_t> klen = {9, 7};

    // Build the dense kvm = o < lens[b][j] per page, with the fork-freeze hole:
    // lane0 page1 (j index 4..7): off0,1 valid, off2 INVALID (hole), off3 invalid.
    std::vector<std::uint8_t> kvm(static_cast<std::size_t>(B) * P_PAGE, 0);
    auto set = [&](int b, int page, int off, std::uint8_t v) {
        kvm[static_cast<std::size_t>(b) * P_PAGE + page * PAGE + off] = v;
    };
    // lane 0: page0 full (4), page1 [0,1] valid + [2]=HOLE + [3] invalid, page2[0].
    for (int o = 0; o < 4; ++o) set(0, 0, o, 1);
    set(0, 1, 0, 1); set(0, 1, 1, 1); set(0, 1, 2, 0); set(0, 1, 3, 0);
    set(0, 2, 0, 1);
    // lane 1: page0 full, page1 [0,1,2] valid (own append), rest invalid.
    for (int o = 0; o < 4; ++o) set(1, 0, o, 1);
    set(1, 1, 0, 1); set(1, 1, 1, 1); set(1, 1, 2, 1);

    // mask_indptr (bytes) = prefix-sum of ceil(klen[b]/8): [0, 2, 3].
    std::vector<std::int32_t> indptr(B + 1, 0);
    for (int b = 0; b < B; ++b)
        indptr[b + 1] = indptr[b] + (static_cast<int>(klen[b]) + 7) / 8;

    const std::vector<std::uint8_t> want = pack_oracle(kvm, klen, indptr, B, P_PAGE);

    // Device round-trip.
    std::uint8_t* d_kvm = nullptr;
    std::uint32_t* d_klen = nullptr;
    std::int32_t* d_indptr = nullptr;
    std::uint32_t* d_qo_indptr = nullptr;
    std::uint8_t* d_packed = nullptr;
    // One query row per lane: qo_indptr = [0, 1, .., B] keeps the classic
    // per-lane bitmap semantics under the [TOTAL_Q, STRIDE] signature.
    std::vector<std::uint32_t> qo_indptr(static_cast<std::size_t>(B) + 1);
    for (int b = 0; b <= B; ++b) qo_indptr[static_cast<std::size_t>(b)] = b;
    CUDA_RT(cudaMalloc(&d_kvm, kvm.size()));
    CUDA_RT(cudaMalloc(&d_klen, klen.size() * 4));
    CUDA_RT(cudaMalloc(&d_indptr, indptr.size() * 4));
    CUDA_RT(cudaMalloc(&d_qo_indptr, qo_indptr.size() * 4));
    CUDA_RT(cudaMalloc(&d_packed, want.size()));
    CUDA_RT(cudaMemcpy(d_kvm, kvm.data(), kvm.size(), cudaMemcpyHostToDevice));
    CUDA_RT(cudaMemcpy(d_klen, klen.data(), klen.size() * 4, cudaMemcpyHostToDevice));
    CUDA_RT(cudaMemcpy(d_indptr, indptr.data(), indptr.size() * 4, cudaMemcpyHostToDevice));
    CUDA_RT(cudaMemcpy(d_qo_indptr, qo_indptr.data(), qo_indptr.size() * 4,
                       cudaMemcpyHostToDevice));
    CUDA_RT(cudaMemset(d_packed, 0, want.size()));

    pie_cuda_driver::kernels::launch_pack_dense_mask(
        d_kvm, d_klen, d_qo_indptr, d_indptr, d_packed, B, P_PAGE, nullptr);
    CUDA_RT(cudaDeviceSynchronize());

    std::vector<std::uint8_t> got(want.size(), 0);
    CUDA_RT(cudaMemcpy(got.data(), d_packed, want.size(), cudaMemcpyDeviceToHost));

    CHECK(got == want, "packed bitmap != oracle on fork-freeze geometry");

    // Explicit freeze semantics: lane 0 bit 6 (page1 off2, j=4*1+2=6) MUST be 0
    // (frozen: not the sibling's tail token); lane 1 bit 6 MUST be 1 (own append).
    auto bit = [&](int b, int j) {
        const std::uint8_t* row = got.data() + indptr[b];
        return (row[j / 8] >> (j % 8)) & 1u;
    };
    CHECK(bit(0, 6) == 0, "lane 0 must NOT see the sibling's tail token (bit 6)");
    CHECK(bit(1, 6) == 1, "lane 1 (heir) must see its own append (bit 6)");
    // Shared prefix visible: lane 0 bits 0..5 valid (page0 full + page1 off0,1).
    for (int j = 0; j < 6; ++j) CHECK(bit(0, j) == 1, "lane 0 shared-prefix bit");
    CHECK(bit(0, 8) == 1, "lane 0 page2 off0 (j=8) valid");

    cudaFree(d_kvm); cudaFree(d_klen); cudaFree(d_indptr); cudaFree(d_packed);
}

}  // namespace

int main() {
    test_fork_freeze();
    if (g_failures) {
        std::fprintf(stderr, "pack_dense_mask: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("pack_dense_mask: fork-freeze kvm packing OK\n");
    return 0;
}
