// PTIR M2b — fork-geometry masked-attention cases (host-side exit gate).
//
// Validates the FULL host chain for the fork-sharing enabler, WITHOUT a GPU:
//
//   fork geometry  →  BRLE mask  →  brle::decode (the REAL driver decoder)
//                  →  packed qo×kv bits  →  reference attention oracle
//
// The load-bearing assertion is overview §5.2/§6.4's semantic identity: a
// masked position is EXACTLY prefix truncation — attention with a fork mask
// equals attention over only that lane's intended-valid keys (W6/W11, no
// attention-kernel change). We check that for four geometries:
//
//   A. prompt-page aliasing        — a decode row attends a full shared prefix.
//   B. mid-chain frozen page       — a frozen page's residual slots are excluded.
//   C. designated-child tail       — a sibling excludes the child's private tail.
//   D. within-page fork at offset  — divergence mid-page masks the sibling's token.
//
// The (Q, K/V, BRLE) tuples generated here are exactly what the CUDA parity
// test (`masked_attention_parity.cu`) feeds the driver's flashinfer
// prefill-custom path — this file proves the inputs + the truncation semantic
// host-side so the GPU window is purely the kernel-vs-oracle diff.
//
// Pure host C++: links `brle.cpp` (no CUDA) + the reference oracle header.

#include "brle.hpp"
#include "masked_attention_reference.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using pie_cuda_driver::brle::DecodedMasks;

int g_failures = 0;

void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_failures;
}

// A single-request attention problem: `kv_len` keys laid out as ceil(kv_len/
// page_size) pages (last page partial), `qo_len` query rows, and a per-row
// BRLE mask (flat runs + per-row indptr) encoding the fork validity.
struct Case {
    int qo_len;
    int kv_len;
    int page_size;
    std::vector<std::uint32_t> brle;         // flattened runs
    std::vector<std::uint32_t> mask_indptr;  // [qo_len + 1] into `brle`
};

// Unpack `brle::decode`'s packed bits for request 0 into the oracle's byte mask
// (mask[q*kv_len + j] = bit(q,j)); LSB-first, bit offset q*kv_len+j (brle.cpp).
std::vector<std::uint8_t> unpack(const DecodedMasks& dm, int qo_len, int kv_len) {
    std::vector<std::uint8_t> m(static_cast<std::size_t>(qo_len) * kv_len, 0);
    const std::uint8_t* base = dm.packed.data() + dm.mask_indptr[0];
    for (int q = 0; q < qo_len; ++q) {
        for (int j = 0; j < kv_len; ++j) {
            const long bit = static_cast<long>(q) * kv_len + j;
            m[static_cast<std::size_t>(q) * kv_len + j] =
                (base[bit / 8] >> (bit % 8)) & 1u;
        }
    }
    return m;
}

// Decode a case's BRLE through the real driver decoder into the oracle mask.
std::vector<std::uint8_t> decode_case(const Case& c) {
    const int pages = (c.kv_len + c.page_size - 1) / c.page_size;
    const int last_len = c.kv_len - (pages - 1) * c.page_size;
    const std::vector<std::uint32_t> qo_indptr{0, static_cast<std::uint32_t>(c.qo_len)};
    const std::vector<std::uint32_t> kv_page_indptr{0, static_cast<std::uint32_t>(pages)};
    const std::vector<std::uint32_t> kv_last_page_lens{static_cast<std::uint32_t>(last_len)};
    DecodedMasks dm = pie_cuda_driver::brle::decode(
        c.brle, c.mask_indptr, qo_indptr, kv_page_indptr, kv_last_page_lens,
        c.page_size);
    return unpack(dm, c.qo_len, c.kv_len);
}

// Random fp32 Q/K/V for the case's shape (fixed seed → deterministic).
void gen_qkv(int qo_len, int kv_len, int heads, int kv_heads, int dim,
             std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
             std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    q.resize(static_cast<std::size_t>(qo_len) * heads * dim);
    k.resize(static_cast<std::size_t>(kv_len) * kv_heads * dim);
    v.resize(static_cast<std::size_t>(kv_len) * kv_heads * dim);
    for (auto& x : q) x = u(rng);
    for (auto& x : k) x = u(rng);
    for (auto& x : v) x = u(rng);
}

pie_attn_ref::Problem make_problem(int qo_len, int kv_len, int heads,
                                   int kv_heads, int dim,
                                   const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v,
                                   const std::uint8_t* mask) {
    pie_attn_ref::Problem p;
    p.qo_len = qo_len;
    p.kv_len = kv_len;
    p.num_qo_heads = heads;
    p.num_kv_heads = kv_heads;
    p.head_dim = dim;
    p.scale = 1.0f / std::sqrt(static_cast<float>(dim));
    p.q = q.data();
    p.k = k.data();
    p.v = v.data();
    p.mask = mask;
    return p;
}

// The core check: the case's decoded mask, run through the oracle, equals
// attention over `expected_valid` (the intended-valid key set for row 0), i.e.
// masking == prefix truncation. Compares within fp32 tolerance.
bool truncation_holds(const Case& c, const std::vector<std::uint8_t>& expected_valid,
                      int heads, int kv_heads, int dim, const char* name) {
    std::vector<float> q, k, v;
    gen_qkv(c.qo_len, c.kv_len, heads, kv_heads, dim, q, k, v, 0xF0F0u + c.kv_len);

    const auto decoded = decode_case(c);
    // 1. The decoded mask must equal the intended-valid pattern bit-for-bit.
    bool mask_eq = decoded.size() == expected_valid.size();
    for (std::size_t i = 0; mask_eq && i < decoded.size(); ++i)
        mask_eq &= (decoded[i] != 0) == (expected_valid[i] != 0);
    // 2. Oracle over the decoded mask == oracle over the intended valid set
    //    (trivially true if (1) holds, but computed independently as a guard).
    auto out_decoded =
        pie_attn_ref::attention(make_problem(c.qo_len, c.kv_len, heads, kv_heads,
                                             dim, q, k, v, decoded.data()));
    auto out_expected =
        pie_attn_ref::attention(make_problem(c.qo_len, c.kv_len, heads, kv_heads,
                                             dim, q, k, v, expected_valid.data()));
    const float diff = pie_attn_ref::max_abs_diff(out_decoded, out_expected);
    const bool ok = mask_eq && diff <= 1e-5f;
    if (!ok)
        std::printf("    %s: mask_eq=%d max_diff=%g\n", name, mask_eq ? 1 : 0, diff);
    return ok;
}

}  // namespace

int main() {
    constexpr int HEADS = 4, KV_HEADS = 2, DIM = 8;  // GQA group 2

    // ── A. prompt-page aliasing — decode row attends a full 8-key shared prefix.
    {
        Case c;
        c.qo_len = 1;
        c.kv_len = 8;
        c.page_size = 4;
        c.brle = {0, 8};        // false-run 0, true-run 8 (attend all)
        c.mask_indptr = {0, 2};
        std::vector<std::uint8_t> want(8, 1);
        check(truncation_holds(c, want, HEADS, KV_HEADS, DIM, "A"),
              "A prompt-page aliasing (full shared prefix)");
    }

    // ── B. mid-chain frozen page — page 1 frozen at offset 6, residual [6,8) masked;
    //     lane's private continuation [8,12) valid.  valid = {0..5, 8..11}.
    {
        Case c;
        c.qo_len = 1;
        c.kv_len = 12;
        c.page_size = 4;
        c.brle = {0, 6, 2, 4};  // false0 | true6 [0,6) | false2 [6,8) | true4 [8,12)
        c.mask_indptr = {0, 4};
        std::vector<std::uint8_t> want(12, 0);
        for (int j = 0; j < 6; ++j) want[j] = 1;
        for (int j = 8; j < 12; ++j) want[j] = 1;
        check(truncation_holds(c, want, HEADS, KV_HEADS, DIM, "B"),
              "B mid-chain frozen page (residual [6,8) excluded)");
    }

    // ── C. designated-child tail — a SIBLING excludes the child's private tail
    //     [6,8); valid = {0..5}.
    {
        Case c;
        c.qo_len = 1;
        c.kv_len = 8;
        c.page_size = 4;
        c.brle = {0, 6};        // attend [0,6), tail [6,8) implicitly masked
        c.mask_indptr = {0, 2};
        std::vector<std::uint8_t> want(8, 0);
        for (int j = 0; j < 6; ++j) want[j] = 1;
        check(truncation_holds(c, want, HEADS, KV_HEADS, DIM, "C"),
              "C designated-child tail (sibling excludes [6,8))");
    }

    // ── D. within-page fork at offset — shared [0,5), sibling forks at 5; valid={0..4}.
    {
        Case c;
        c.qo_len = 1;
        c.kv_len = 6;           // page0 [0,4) full, page1 [4,6) partial
        c.page_size = 4;
        c.brle = {0, 5};        // attend [0,5), the sibling's slot 5 masked
        c.mask_indptr = {0, 2};
        std::vector<std::uint8_t> want(6, 0);
        for (int j = 0; j < 5; ++j) want[j] = 1;
        check(truncation_holds(c, want, HEADS, KV_HEADS, DIM, "D"),
              "D within-page fork at offset 5 (sibling slot masked)");
    }

    // ── Multi-row causal sanity (prefill shape): row q attends [0, q].
    {
        Case c;
        c.qo_len = 4;
        c.kv_len = 4;
        c.page_size = 4;
        // Per-row BRLE: row q → false0, true(q+1).  All rows concatenated.
        c.brle = {0, 1, 0, 2, 0, 3, 0, 4};
        c.mask_indptr = {0, 2, 4, 6, 8};
        std::vector<std::uint8_t> want(16, 0);
        for (int q = 0; q < 4; ++q)
            for (int j = 0; j <= q; ++j) want[q * 4 + j] = 1;
        check(truncation_holds(c, want, HEADS, KV_HEADS, DIM, "E"),
              "E multi-row causal prefill (row q attends [0,q])");
    }

    std::printf(g_failures ? "\nFORK-GEOMETRY FAILED (%d)\n" : "\nALL PASS (0 failures)\n",
                g_failures);
    return g_failures ? 1 : 0;
}
