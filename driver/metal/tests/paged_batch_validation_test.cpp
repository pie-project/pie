#include <cstdio>
#include <string>
#include <vector>

#include "batch_schedule.hpp"

using namespace pie_metal_driver::raw_metal;

namespace {
int pass = 0, fail = 0;
void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++pass : ++fail;
}
}  // namespace

int main() {
    std::printf("[paged batch validation]\n");
    const uint32_t tokens[] = {10, 11};
    const uint32_t qo[] = {0, 1, 2};
    // Request 0 uses pages {7,2}; request 1 shares page 7.  Physical CSR
    // order is intentionally non-monotone and shared.
    const uint32_t pi[] = {0, 2, 3};
    const uint32_t pages[] = {7, 2, 7};
    const uint32_t last[] = {32, 1};
    const uint32_t slots[] = {0, 1};
    const uint8_t flags[] = {0, 1};
    BatchSchedule s = build_batch_schedule(tokens, 2, qo, pi, last, slots, flags, 3, 32);
    const std::vector<uint32_t> pos = {31, 0};
    const std::vector<uint32_t> page_vec(pages, pages + 3);
    std::vector<uint32_t> wp = {7, 7}, wo = {31, 0};
    std::string err;
    expect(validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err),
           "ordered physical CSR/shared prefix and matching explicit writes validate (" + err + ")");

    wp[1] = 2;
    expect(!validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err) &&
               err.find("disagrees") != std::string::npos,
           "write page that disagrees with CSR is rejected before dispatch");
    wp[1] = 7;
    wo[0] = 32;
    expect(!validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err) &&
               err.find("exceeds") != std::string::npos,
           "out-of-page write offset is rejected before dispatch");
    wo[0] = 31;
    std::vector<uint32_t> bad_pages = {7, 9, 7};
    expect(!validate_paged_batch(s, pos, bad_pages, wp, wo, 8, 2, &err) &&
               err.find("physical page") != std::string::npos,
           "out-of-pool CSR page is rejected before dispatch");

    BatchSchedule over_cap = s;
    over_cap.N = 65;
    expect(!validate_paged_batch_capacity(over_cap, 64, 4, &err) &&
               err.find("capacity") != std::string::npos,
           "over-cap (>64 token) paged prompt rejects before command encoding");

    std::printf("\n==== paged_batch_validation_test: %d passed, %d failed ====\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
