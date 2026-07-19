// Unit tests for packed-mask `is_pure_causal` + `decode`.
//
// Self-contained: builds against the same `brle.cpp` translation unit
// the driver uses; no test framework — failures abort with a message
// and a non-zero exit code so CTest picks them up.
//
// Test cases exercise the wire-format shapes the runtime ships:
//   * pure-causal single request (most common)
//   * pure-causal multi-request batch
//   * non-causal packed rows (jacobi-style alternating)
//   * empty masks and kv_len truncation

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "batch/brle.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                 \
    do {                                                            \
        if (!(cond)) {                                              \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",               \
                         __FILE__, __LINE__, #cond);                \
            ++g_failures;                                           \
        }                                                           \
    } while (0)

#define CHECK_EQ(a, b)                                              \
    do {                                                            \
        const auto _a = (a);                                        \
        const auto _b = (b);                                        \
        if (!(_a == _b)) {                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s == %s "           \
                         "(got %lld vs %lld)\n",                    \
                         __FILE__, __LINE__, #a, #b,                \
                         static_cast<long long>(_a),                \
                         static_cast<long long>(_b));               \
            ++g_failures;                                           \
        }                                                           \
    } while (0)

std::vector<std::uint32_t> causal_row(int p, int kv_len) {
    std::vector<std::uint32_t> words(
        static_cast<std::size_t>(kv_len + 31) / 32, 0);
    for (int key = 0; key <= p; ++key) {
        words[static_cast<std::size_t>(key) / 32] |=
            1u << (key % 32);
    }
    return words;
}

// Concatenate packed rows into the ABI's flat word buffer + row indptr.
struct FlatMask {
    std::vector<std::uint32_t> words;
    std::vector<std::uint32_t> indptr;
};
FlatMask flatten(const std::vector<std::vector<std::uint32_t>>& rows) {
    FlatMask out;
    out.indptr.push_back(0);
    for (const auto& r : rows) {
        out.words.insert(out.words.end(), r.begin(), r.end());
        out.indptr.push_back(static_cast<std::uint32_t>(out.words.size()));
    }
    return out;
}

// Read a single bit from a row of the packed bitmap, matching the
// LSB-first-within-byte convention `brle::decode` writes.
bool bit(const std::uint8_t* row_base, std::int64_t row_off,
         int q, int kv_len, int k) {
    const std::int64_t bit_off = row_off + static_cast<std::int64_t>(q) * kv_len + k;
    return (row_base[bit_off / 8] >> (bit_off % 8)) & 1u;
}

void test_empty_batch() {
    namespace brle = pie_cuda_driver::brle;
    std::vector<std::uint32_t> qo_indptr = {0};
    std::vector<std::uint32_t> kv_page_indptr = {0};
    std::vector<std::uint32_t> kv_last_page_lens;
    auto decoded = brle::decode({}, {},
                                std::span<const std::uint32_t>(qo_indptr),
                                std::span<const std::uint32_t>(kv_page_indptr),
                                std::span<const std::uint32_t>(kv_last_page_lens),
                                /*page_size=*/16);
    CHECK(decoded.packed.empty());
    CHECK_EQ(decoded.mask_indptr.size(), 1);
    CHECK_EQ(decoded.mask_indptr[0], 0);
}

void test_pure_causal_single_request() {
    namespace brle = pie_cuda_driver::brle;
    constexpr int page_size = 4;
    constexpr int kv_len = 5;             // 1 full page + 1 token
    constexpr int qo_len = 3;             // prefill stride 3

    // Three causal rows over 5 keys.
    std::vector<std::vector<std::uint32_t>> rows;
    for (int q = 0; q < qo_len; ++q) {
        rows.push_back(causal_row(q + (kv_len - qo_len), kv_len));
    }
    auto flat = flatten(rows);

    std::vector<std::uint32_t> qo_indptr     = {0u, qo_len};
    std::vector<std::uint32_t> kv_page_indptr = {0u, 2u};   // 2 pages
    std::vector<std::uint32_t> kv_last_lpl    = {1u};       // last page = 1 token

    auto causal = brle::is_pure_causal(
        std::span<const std::uint32_t>(flat.words),
        std::span<const std::uint32_t>(flat.indptr),
        std::span<const std::uint32_t>(qo_indptr),
        std::span<const std::uint32_t>(kv_page_indptr),
        std::span<const std::uint32_t>(kv_last_lpl),
        page_size);
    CHECK(causal);

    auto decoded = brle::decode(
        std::span<const std::uint32_t>(flat.words),
        std::span<const std::uint32_t>(flat.indptr),
        std::span<const std::uint32_t>(qo_indptr),
        std::span<const std::uint32_t>(kv_page_indptr),
        std::span<const std::uint32_t>(kv_last_lpl),
        page_size);

    // Bytes per request: ceil(qo_len * kv_len / 8) = ceil(15/8) = 2.
    CHECK_EQ(decoded.mask_indptr.size(), 2);
    CHECK_EQ(decoded.mask_indptr[1], 2);
    CHECK_EQ(decoded.packed.size(), 2);

    // Bit grid: row q attends to key k iff k <= (kv_len - qo_len + q).
    const std::uint8_t* base = decoded.packed.data();
    for (int q = 0; q < qo_len; ++q) {
        const int abs_q = (kv_len - qo_len) + q;
        for (int k = 0; k < kv_len; ++k) {
            const bool expected = (k <= abs_q);
            const bool got = bit(base, 0, q, kv_len, k);
            if (got != expected) {
                std::fprintf(stderr,
                    "FAIL pure_causal q=%d k=%d expected=%d got=%d\n",
                    q, k, expected, got);
                ++g_failures;
            }
        }
    }
}

void test_pure_causal_multi_request() {
    namespace brle = pie_cuda_driver::brle;
    constexpr int page_size = 4;
    // Two requests: r0 has qo=2, kv=2; r1 has qo=1, kv=4.
    std::vector<std::vector<std::uint32_t>> rows = {
        causal_row(0, 2), causal_row(1, 2),
        causal_row(3, 4),
    };
    auto flat = flatten(rows);
    std::vector<std::uint32_t> qo_indptr      = {0u, 2u, 3u};
    std::vector<std::uint32_t> kv_page_indptr = {0u, 1u, 2u};
    std::vector<std::uint32_t> kv_last_lpl    = {2u, 4u};

    CHECK(brle::is_pure_causal(
        std::span<const std::uint32_t>(flat.words),
        std::span<const std::uint32_t>(flat.indptr),
        std::span<const std::uint32_t>(qo_indptr),
        std::span<const std::uint32_t>(kv_page_indptr),
        std::span<const std::uint32_t>(kv_last_lpl),
        page_size));
}

void test_causal_prefix_with_reserved_pages() {
    namespace brle = pie_cuda_driver::brle;
    constexpr int page_size = 4;
    auto flat = flatten({
        causal_row(4, 12),
        causal_row(5, 12),
    });
    std::vector<std::uint32_t> qo_indptr = {0u, 2u};
    std::vector<std::uint32_t> kv_page_indptr = {0u, 3u};
    std::vector<std::uint32_t> kv_last_lpl = {4u};
    std::vector<std::uint32_t> lengths;

    CHECK(brle::causal_prefix_lengths(
        flat.words, flat.indptr, qo_indptr, kv_page_indptr,
        kv_last_lpl, page_size, lengths));
    CHECK_EQ(lengths.size(), 1);
    CHECK_EQ(lengths[0], 6);
    CHECK(!brle::is_pure_causal(
        flat.words, flat.indptr, qo_indptr, kv_page_indptr,
        kv_last_lpl, page_size));
}

void test_non_causal_alternating() {
    namespace brle = pie_cuda_driver::brle;
    constexpr int page_size = 8;
    // Single request, qo=1, kv=4. Mask = [F T F T] — alternating, not
    // a causal triangle.
    std::vector<std::uint32_t> runs = {0b1010u};
    std::vector<std::uint32_t> indptr = {0u, 1u};
    std::vector<std::uint32_t> qo_indptr      = {0u, 1u};
    std::vector<std::uint32_t> kv_page_indptr = {0u, 1u};
    std::vector<std::uint32_t> kv_last_lpl    = {4u};

    CHECK(!brle::is_pure_causal(runs, indptr, qo_indptr,
                                kv_page_indptr, kv_last_lpl, page_size));

    auto decoded = brle::decode(runs, indptr, qo_indptr,
                                kv_page_indptr, kv_last_lpl, page_size);
    CHECK_EQ(decoded.packed.size(), 1);  // ceil(4/8) = 1 byte
    // bit 0 = false, bit 1 = true, bit 2 = false, bit 3 = true.
    const std::uint8_t b = decoded.packed[0];
    CHECK(!(b & (1u << 0)));
    CHECK( (b & (1u << 1)));
    CHECK(!(b & (1u << 2)));
    CHECK( (b & (1u << 3)));
}

void test_runs_truncated_at_kv_len() {
    namespace brle = pie_cuda_driver::brle;
    constexpr int page_size = 4;
    // Single request: qo=1, kv=3 (1 page = 4 - 1 unused slot).
    // The packed row has ten set bits; the decoder clips to kv=3.
    std::vector<std::uint32_t> runs = {0x3ffu};
    std::vector<std::uint32_t> indptr = {0u, 1u};
    std::vector<std::uint32_t> qo_indptr      = {0u, 1u};
    std::vector<std::uint32_t> kv_page_indptr = {0u, 1u};
    std::vector<std::uint32_t> kv_last_lpl    = {3u};

    auto decoded = brle::decode(runs, indptr, qo_indptr,
                                kv_page_indptr, kv_last_lpl, page_size);
    CHECK_EQ(decoded.packed.size(), 1);
    const std::uint8_t b = decoded.packed[0];
    // Expect bits 0,1,2 set; bit 3 clear (no padding into bit 3).
    CHECK( (b & 0b0000'0111));
    CHECK(!(b & 0b1111'1000));
}

}  // namespace

int main() {
    test_empty_batch();
    test_pure_causal_single_request();
    test_pure_causal_multi_request();
    test_causal_prefix_with_reserved_pages();
    test_non_causal_alternating();
    test_runs_truncated_at_kv_len();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED (%d checks)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
