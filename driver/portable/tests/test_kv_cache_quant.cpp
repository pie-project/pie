#include "kv_cache_quant.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using pie_portable_driver::KvCacheQuantScheme;
using pie_portable_driver::kv_cache_quant_format_from_string;
using pie_portable_driver::qdq_kv_row;

namespace {

void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

void require_close(float actual, float expected, float tol, const std::string& msg) {
    if (std::fabs(actual - expected) > tol) {
        throw std::runtime_error(
            msg + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void test_parse() {
    const char* valid[] = {
        "auto",
        "bf16",
        "bfloat16",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "fp8_per_token_head",
        "fp4_e2m1",
        "nvfp4",
    };
    for (const char* dtype : valid) {
        (void)kv_cache_quant_format_from_string(dtype);
    }
    bool threw = false;
    try {
        (void)kv_cache_quant_format_from_string("not_a_dtype");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "invalid dtype should throw");
}

void test_native_noop() {
    auto fmt = kv_cache_quant_format_from_string("auto");
    std::vector<float> row = {0.1f, -0.2f, 1.25f, -3.5f};
    auto before = row;
    qdq_kv_row(row.data(), 2, 2, fmt);
    require(row == before, "native qdq should be a no-op");
}

void test_int8_per_token_head() {
    auto fmt = kv_cache_quant_format_from_string("int8_per_token_head");
    std::vector<float> row = {1.0f, -0.5f, 0.25f, 3.0f, -1.5f, 0.75f};
    qdq_kv_row(row.data(), 2, 3, fmt);

    const float scale0 = 1.0f / 127.0f;
    require_close(row[0], 127.0f * scale0, 1e-6f, "int8 head0 d0");
    require_close(row[1], -64.0f * scale0, 1e-6f, "int8 head0 d1");

    const float scale1 = 3.0f / 127.0f;
    require_close(row[3], 127.0f * scale1, 1e-6f, "int8 head1 d0");
    require_close(row[4], -64.0f * scale1, 1e-6f, "int8 head1 d1");
}

void test_fp8_and_fp4_change_values() {
    std::vector<float> row = {0.1234f, -0.5678f, 1.2345f, -2.3456f};
    auto fp8 = row;
    auto fmt8 = kv_cache_quant_format_from_string("fp8_e4m3");
    qdq_kv_row(fp8.data(), 1, 4, fmt8);
    bool any_changed = false;
    for (std::size_t i = 0; i < row.size(); ++i) {
        any_changed = any_changed || std::fabs(row[i] - fp8[i]) > 1e-5f;
    }
    require(any_changed, "fp8 qdq should change at least one value");

    auto fp4 = row;
    auto fmt4 = kv_cache_quant_format_from_string("nvfp4");
    require(fmt4.scheme == KvCacheQuantScheme::Fp4Block, "nvfp4 should parse as fp4 block");
    qdq_kv_row(fp4.data(), 1, 4, fmt4);
    any_changed = false;
    for (std::size_t i = 0; i < row.size(); ++i) {
        any_changed = any_changed || std::fabs(row[i] - fp4[i]) > 1e-5f;
    }
    require(any_changed, "fp4 qdq should change at least one value");
}

}  // namespace

int main() {
    try {
        test_parse();
        test_native_noop();
        test_int8_per_token_head();
        test_fp8_and_fp4_change_values();
        std::puts("portable kv_cache_quant ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "portable kv_cache_quant failed: %s\n", e.what());
        return 1;
    }
}
