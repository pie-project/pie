#include <cstdint>
#include <cstdio>

#include "pipeline/program_identity.hpp"

using pie_cuda_driver::pipeline::ProgramSetIdentityFold;
using pie_cuda_driver::pipeline::compiled_stage_identity;
using pie_cuda_driver::pipeline::program_extent_bucket;
using pie_native::ptir::plan::Region;
using pie_native::ptir::plan::StagePlan;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

std::uint64_t folded(
    std::uint64_t first,
    std::uint8_t first_bucket,
    std::uint64_t second,
    std::uint8_t second_bucket,
    bool duplicate_first = false) {
    ProgramSetIdentityFold fold;
    fold.add(first, first_bucket);
    fold.add(second, second_bucket);
    if (duplicate_first) fold.add(first, first_bucket);
    return fold.finish();
}

}  // namespace

int main() {
    StagePlan first;
    first.stage = PTIR_STAGE_EPILOGUE;
    first.signature = {'s', 't', 'a', 'g', 'e', '-', 'a'};
    first.fused.regions = {
        Region{false, 0, PTIR_SCHEDULE_ONE_CTA_PER_ROW, {0, 1}, {}, {}, {}},
        Region{true, PTIR_LIBRARY_TOP_K, PTIR_SCHEDULE_LIBRARY, {2}, {}, {}, {}},
    };
    StagePlan same = first;
    StagePlan different_signature = first;
    different_signature.signature.back() = 'b';
    StagePlan different_region = first;
    different_region.fused.regions[0].nodes.push_back(3);

    const std::uint64_t first_id = compiled_stage_identity(first);
    const std::uint64_t same_id = compiled_stage_identity(same);
    const std::uint64_t signature_id =
        compiled_stage_identity(different_signature);
    const std::uint64_t region_id =
        compiled_stage_identity(different_region);
    expect(first_id != 0 && first_id == same_id,
           "immutable stage identity is stable");
    expect(first_id != signature_id,
           "stage identity includes canonical signature bytes");
    expect(first_id != region_id,
           "stage identity includes fused region structure");
    expect(
        program_extent_bucket(0) == 0 &&
            program_extent_bucket(1) == 0 &&
            program_extent_bucket(2) == 1 &&
            program_extent_bucket(3) == 2 &&
            program_extent_bucket(4) == 2,
        "row buckets preserve the compiled schedule classes");

    const std::uint64_t ordered = folded(first_id, 1, region_id, 2);
    const std::uint64_t reversed = folded(region_id, 2, first_id, 1);
    const std::uint64_t duplicated =
        folded(first_id, 1, region_id, 2, true);
    const std::uint64_t rebucketed = folded(first_id, 2, region_id, 2);
    expect(ordered != 0 && ordered == reversed,
           "program-set fold is order independent");
    expect(ordered != duplicated,
           "program-set fold includes exact stage multiplicity");
    expect(ordered != rebucketed,
           "program-set fold includes per-program row buckets");
    expect(ProgramSetIdentityFold{}.finish() == 0,
           "empty program set preserves the non-PTIR identity");

    std::printf("program_identity_test: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
